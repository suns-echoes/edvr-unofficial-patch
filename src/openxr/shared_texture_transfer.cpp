#include "shared_texture_transfer.h"

#include "immediate_executor.h"
#include "gpu_work_observer.h"
#include "producer_gpu_timing.h"
#include "native_trace.h"
#include <d3d11_4.h>
#include <dxgi1_2.h>
#include <wrl/client.h>
#include <exception>
#include <thread>
#include <utility>

namespace edvr::openxr {
namespace {

using Microsoft::WRL::ComPtr;

enum class PixelFamily { RGBA8, BGRA8 };

struct FormatChoice {
  PixelFamily family;
  DXGI_FORMAT shared;
  DXGI_FORMAT privateFormat;
};

bool formatChoice(DXGI_FORMAT format, FormatChoice& choice) noexcept {
  switch (format) {
    case DXGI_FORMAT_R8G8B8A8_TYPELESS:
    case DXGI_FORMAT_R8G8B8A8_UNORM:
    case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
      choice = {PixelFamily::RGBA8, DXGI_FORMAT_R8G8B8A8_UNORM,
                DXGI_FORMAT_R8G8B8A8_TYPELESS};
      return true;
    case DXGI_FORMAT_B8G8R8A8_TYPELESS:
    case DXGI_FORMAT_B8G8R8A8_UNORM:
    case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
      choice = {PixelFamily::BGRA8, DXGI_FORMAT_B8G8R8A8_UNORM,
                DXGI_FORMAT_B8G8R8A8_TYPELESS};
      return true;
    default:
      return false;
  }
}

bool sameLuid(const LUID& a, const LUID& b) noexcept {
  return a.HighPart == b.HighPart && a.LowPart == b.LowPart;
}

bool sameDevice(ID3D11Device* first, ID3D11Device* second) noexcept {
  if (!first || !second) return false;
  ComPtr<IUnknown> firstIdentity, secondIdentity;
  if (FAILED(first->QueryInterface(IID_PPV_ARGS(&firstIdentity))) ||
      FAILED(second->QueryInterface(IID_PPV_ARGS(&secondIdentity)))) return false;
  return firstIdentity.Get() == secondIdentity.Get();
}

bool adapterLuid(ID3D11Device* device, LUID& luid) noexcept {
  ComPtr<IDXGIDevice> dxgiDevice;
  ComPtr<IDXGIAdapter> adapter;
  DXGI_ADAPTER_DESC description{};
  if (!device || FAILED(device->QueryInterface(IID_PPV_ARGS(&dxgiDevice))) ||
      FAILED(dxgiDevice->GetAdapter(&adapter)) || !adapter ||
      FAILED(adapter->GetDesc(&description))) return false;
  luid = description.AdapterLuid;
  return true;
}

bool validSource(const D3D11_TEXTURE2D_DESC& description) noexcept {
  return description.Width != 0 && description.Height != 0 &&
      description.Width <= D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION &&
      description.Height <= D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION &&
      description.MipLevels == 1 && description.ArraySize == 1 &&
      description.SampleDesc.Count == 1 && description.SampleDesc.Quality == 0 &&
      description.Usage == D3D11_USAGE_DEFAULT && description.CPUAccessFlags == 0;
}

HRESULT timeoutResult() noexcept {
  return HRESULT_FROM_WIN32(ERROR_TIMEOUT);
}

// How long the one-time fence probe may wait for each direction's first
// signal. The signal queues behind the GPU work already submitted on that
// device, so this is generous; a miss only selects the keyed-mutex handoff.
constexpr DWORD kFenceProbeMs = 2000;

// Polls a fence on the owner thread until it reaches value. UINT64_MAX is a
// removed device, never a reached value.
bool fenceReached(ID3D11Fence* fence, UINT64 value, DWORD timeoutMs) noexcept {
  const ULONGLONG started = GetTickCount64();
  for (;;) {
    const UINT64 completed = fence->GetCompletedValue();
    if (completed != UINT64_MAX && completed >= value) return true;
    if (completed == UINT64_MAX || GetTickCount64() - started >= timeoutMs) return false;
    Sleep(1);
  }
}

HRESULT shareFence(ID3D11Fence* fence, HANDLE& handle) noexcept {
  handle = nullptr;
  const HRESULT result = fence->CreateSharedHandle(nullptr, GENERIC_ALL, nullptr, &handle);
  if (result == S_OK && !handle) return E_FAIL;
  return result;
}

}  // namespace

struct SharedTextureTransfer::Impl final {
  enum class Stage { None, ConsumerAcquire };

  explicit Impl() : owner(std::this_thread::get_id()) {}

  bool onOwner() const noexcept {
    return owner == std::this_thread::get_id();
  }

  bool fenced() const noexcept {
    return readyOnProducer && readyOnConsumer && freeOnConsumer && freeOnProducer;
  }

  bool hasResources() const noexcept {
    return sharedProducer && sharedConsumer && privateTexture && consumerEvent &&
        (fenced() || (producerMutex && consumerMutex));
  }

  void clearResources() noexcept {
    consumerEvent.Reset();
    consumerMutex.Reset();
    producerMutex.Reset();
    freeOnProducer.Reset();
    freeOnConsumer.Reset();
    readyOnConsumer.Reset();
    readyOnProducer.Reset();
    fenceValue = 0;
    privateTexture.Reset();
    sharedConsumer.Reset();
    sharedProducer.Reset();
    sourceWidth = sourceHeight = 0;
    sourceFamily = PixelFamily::RGBA8;
  }

  HRESULT validateInitialize(ID3D11Device* producerDevice,
                             ID3D11Device* consumerDevice,
                             ImmediateExecutor* executor) noexcept {
    if (!onOwner() || initialized || pendingStage != Stage::None || faulted ||
        !producerDevice || !consumerDevice || !executor ||
        sameDevice(producerDevice, consumerDevice) ||
        (producerDevice->GetCreationFlags() & D3D11_CREATE_DEVICE_SINGLETHREADED) ||
        (consumerDevice->GetCreationFlags() & D3D11_CREATE_DEVICE_SINGLETHREADED))
      return E_INVALIDARG;

    LUID consumerLuid{};
    if (!adapterLuid(consumerDevice, consumerLuid)) return E_NOINTERFACE;

    ComPtr<ID3D11DeviceContext> producerImmediate;
    ComPtr<ID3D11Device5> producer5;
    ComPtr<ID3D11DeviceContext4> producerImmediate4;
    LUID producerLuid{};
    bool producerValid = false;
    bool invoked = false;
    try {
      invoked = executor->invoke([&] {
        producerValid = adapterLuid(producerDevice, producerLuid);
        producerDevice->GetImmediateContext(&producerImmediate);
        producerValid = producerValid && producerImmediate &&
            sameLuid(producerLuid, consumerLuid);
        if (producerValid &&
            (FAILED(producerDevice->QueryInterface(IID_PPV_ARGS(&producer5))) ||
             FAILED(producerImmediate.As(&producerImmediate4)))) {
          producer5.Reset();
          producerImmediate4.Reset();
        }
      });
    } catch (...) {}
    if (!invoked || !producerValid) return invoked ? E_INVALIDARG : E_ABORT;

    ComPtr<ID3D11DeviceContext> consumerImmediate;
    consumerDevice->GetImmediateContext(&consumerImmediate);
    if (!consumerImmediate) return E_NOINTERFACE;

    ComPtr<ID3D11Device1> consumer1;
    if (FAILED(consumerDevice->QueryInterface(IID_PPV_ARGS(&consumer1))) || !consumer1)
      return E_NOINTERFACE;

    ComPtr<ID3D11Device5> consumer5;
    ComPtr<ID3D11DeviceContext4> consumerImmediate4;
    if (FAILED(consumerDevice->QueryInterface(IID_PPV_ARGS(&consumer5))) ||
        FAILED(consumerImmediate.As(&consumerImmediate4))) {
      consumer5.Reset();
      consumerImmediate4.Reset();
    }

    producer = producerDevice;
    consumer = consumerDevice;
    producerContext = std::move(producerImmediate);
    consumerContext = std::move(consumerImmediate);
    producerExecutor = executor;
    consumerDevice1 = std::move(consumer1);
    producerDevice5 = std::move(producer5);
    consumerDevice5 = std::move(consumer5);
    producerContext4 = std::move(producerImmediate4);
    consumerContext4 = std::move(consumerImmediate4);
    fencesRefused = !(producerDevice5 && consumerDevice5 && producerContext4 && consumerContext4);
    syncReported = false;
    initialized = true;
    return S_OK;
  }

  HRESULT createResources(const D3D11_TEXTURE2D_DESC& sourceDescription,
                          const FormatChoice& choice, DWORD timeoutMs) noexcept {
    if (hasResources() && sourceWidth == sourceDescription.Width &&
        sourceHeight == sourceDescription.Height && sourceFamily == choice.family)
      return S_OK;

    if (hasResources()) {
      const HRESULT drained = drainConsumer(timeoutMs);
      if (drained != S_OK) return drained;
    }

    if (!fencesRefused) {
      const HRESULT fencedResult = createFencedResources(sourceDescription, choice);
      if (fencedResult == S_OK) {
        reportSync("fence", S_OK);
        return S_OK;
      }
      if (fencedResult == E_ABORT) return E_ABORT;
      // Shared fences are missing or do not cross the two devices: keep the
      // keyed-mutex handoff for the rest of this transfer's life.
      fencesRefused = true;
      reportSync("keyed_mutex", fencedResult);
    } else {
      reportSync("keyed_mutex", E_NOINTERFACE);
    }
    return createKeyedResources(sourceDescription, choice);
  }

  void reportSync(const char* mode, HRESULT why) noexcept {
    if (syncReported) return;
    syncReported = true;
    nativeTracePrintf("native_transfer_sync,mode=%s,fence_hr=%08lx\n", mode,
                      static_cast<unsigned long>(why));
  }

  // GPU-side handoff: the producer queues Signal(ready, n) after its copy and
  // the consumer queues Wait(ready, n) before its own; the free fence runs the
  // other way so the producer's next copy waits on the GPU for the consumer's
  // read. Neither CPU thread blocks. Under DXVK, IDXGIKeyedMutex::ReleaseSync
  // waits on the CPU for all of the producer's queued GPU work, which held the
  // game's render thread for most of a frame (native_submit_phases
  // producer_release, Proton 2026-11-06). A probe proves both fences cross the
  // two devices before any frame relies on them.
  HRESULT createFencedResources(const D3D11_TEXTURE2D_DESC& sourceDescription,
                                const FormatChoice& choice) noexcept {
    D3D11_TEXTURE2D_DESC sharedDescription = sourceDescription;
    sharedDescription.Format = choice.shared;
    sharedDescription.Usage = D3D11_USAGE_DEFAULT;
    // Copy destinations only; see createKeyedResources for why there is no RTV.
    sharedDescription.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    sharedDescription.CPUAccessFlags = 0;
    sharedDescription.MiscFlags = D3D11_RESOURCE_MISC_SHARED |
        D3D11_RESOURCE_MISC_SHARED_NTHANDLE;
    D3D11_TEXTURE2D_DESC privateDescription = sharedDescription;
    privateDescription.Format = choice.privateFormat;
    privateDescription.MiscFlags = 0;

    ComPtr<ID3D11Texture2D> nextSharedProducer;
    ComPtr<ID3D11Fence> nextReadyOnProducer;
    HANDLE textureHandle = nullptr, readyHandle = nullptr;
    HRESULT producerResult = E_FAIL;
    bool callbackRan = false;
    try {
      (void)producerExecutor->invoke([&] {
        callbackRan = true;
        producerResult = producer->CreateTexture2D(&sharedDescription, nullptr,
                                                   &nextSharedProducer);
        if (producerResult != S_OK) return;
        ComPtr<IDXGIResource1> resource;
        producerResult = nextSharedProducer.As(&resource);
        if (producerResult != S_OK) return;
        producerResult = resource->CreateSharedHandle(nullptr,
            DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE,
            nullptr, &textureHandle);
        if (producerResult != S_OK) return;
        producerResult = producerDevice5->CreateFence(0, D3D11_FENCE_FLAG_SHARED,
                                                      IID_PPV_ARGS(&nextReadyOnProducer));
        if (producerResult != S_OK) return;
        producerResult = shareFence(nextReadyOnProducer.Get(), readyHandle);
      });
    } catch (...) {}
    const auto closeHandles = [&] {
      if (textureHandle) CloseHandle(textureHandle);
      if (readyHandle) CloseHandle(readyHandle);
      textureHandle = readyHandle = nullptr;
    };
    if (!callbackRan) { closeHandles(); return E_ABORT; }
    if (producerResult != S_OK) { closeHandles(); return producerResult; }

    ComPtr<ID3D11Texture2D> nextSharedConsumer;
    ComPtr<ID3D11Fence> nextReadyOnConsumer, nextFreeOnConsumer, nextFreeOnProducer;
    ComPtr<ID3D11Texture2D> nextPrivate;
    ComPtr<ID3D11Query> nextEvent;
    HRESULT result = consumerDevice1->OpenSharedResource1(
        textureHandle, IID_PPV_ARGS(&nextSharedConsumer));
    if (result == S_OK)
      result = consumerDevice5->OpenSharedFence(readyHandle, IID_PPV_ARGS(&nextReadyOnConsumer));
    closeHandles();
    if (result == S_OK)
      result = consumerDevice5->CreateFence(0, D3D11_FENCE_FLAG_SHARED,
                                            IID_PPV_ARGS(&nextFreeOnConsumer));
    HANDLE freeHandle = nullptr;
    if (result == S_OK) result = shareFence(nextFreeOnConsumer.Get(), freeHandle);
    if (result == S_OK) {
      HRESULT openResult = E_FAIL;
      bool openRan = false;
      try {
        (void)producerExecutor->invoke([&] {
          openRan = true;
          openResult = producerDevice5->OpenSharedFence(freeHandle,
                                                        IID_PPV_ARGS(&nextFreeOnProducer));
        });
      } catch (...) {}
      result = openRan ? openResult : E_ABORT;
    }
    if (freeHandle) CloseHandle(freeHandle);
    if (result == S_OK)
      result = consumer->CreateTexture2D(&privateDescription, nullptr, &nextPrivate);
    if (result == S_OK) {
      const D3D11_QUERY_DESC queryDescription{D3D11_QUERY_EVENT, 0};
      result = consumer->CreateQuery(&queryDescription, &nextEvent);
    }
    if (result != S_OK) return result;
    if (!nextSharedConsumer || !nextReadyOnConsumer || !nextFreeOnConsumer ||
        !nextFreeOnProducer || !nextPrivate || !nextEvent) return E_FAIL;

    // Probe both directions with value 1. A fence that only looks shared (an
    // import the driver ignored) never reaches 1 on the other device.
    bool signalled = false;
    try {
      (void)producerExecutor->invoke([&] {
        signalled = producerContext4->Signal(nextReadyOnProducer.Get(), 1) == S_OK;
        producerContext->Flush();
      });
    } catch (...) {}
    if (!signalled || !fenceReached(nextReadyOnConsumer.Get(), 1, kFenceProbeMs))
      return DXGI_ERROR_UNSUPPORTED;
    if (consumerContext4->Signal(nextFreeOnConsumer.Get(), 1) != S_OK) return DXGI_ERROR_UNSUPPORTED;
    consumerContext->Flush();
    if (!fenceReached(nextFreeOnProducer.Get(), 1, kFenceProbeMs)) return DXGI_ERROR_UNSUPPORTED;

    clearResources();
    sharedProducer = std::move(nextSharedProducer);
    sharedConsumer = std::move(nextSharedConsumer);
    privateTexture = std::move(nextPrivate);
    readyOnProducer = std::move(nextReadyOnProducer);
    readyOnConsumer = std::move(nextReadyOnConsumer);
    freeOnConsumer = std::move(nextFreeOnConsumer);
    freeOnProducer = std::move(nextFreeOnProducer);
    consumerEvent = std::move(nextEvent);
    fenceValue = 1;
    sourceWidth = sourceDescription.Width;
    sourceHeight = sourceDescription.Height;
    sourceFamily = choice.family;
    return S_OK;
  }

  HRESULT createKeyedResources(const D3D11_TEXTURE2D_DESC& sourceDescription,
                               const FormatChoice& choice) noexcept {
    D3D11_TEXTURE2D_DESC sharedDescription = sourceDescription;
    sharedDescription.Format = choice.shared;
    sharedDescription.Usage = D3D11_USAGE_DEFAULT;
    // These resources are copy destinations, never render targets. Keeping
    // the RTV flag off also excludes the producer allocation from the game's
    // FSS/surface-size hooks, which otherwise rewrite matching RT dimensions.
    sharedDescription.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    sharedDescription.CPUAccessFlags = 0;
    sharedDescription.MiscFlags = D3D11_RESOURCE_MISC_SHARED_NTHANDLE |
        D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX;
    D3D11_TEXTURE2D_DESC privateDescription = sharedDescription;
    privateDescription.Format = choice.privateFormat;
    privateDescription.MiscFlags = 0;

    ComPtr<ID3D11Texture2D> nextSharedProducer;
    ComPtr<IDXGIKeyedMutex> nextProducerMutex;
    HANDLE nextHandle = nullptr;
    HRESULT producerResult = E_FAIL;
    bool callbackRan = false;
    try {
      (void)producerExecutor->invoke([&] {
        callbackRan = true;
        producerResult = producer->CreateTexture2D(&sharedDescription, nullptr,
                                                   &nextSharedProducer);
        if (producerResult != S_OK) return;
        producerResult = nextSharedProducer.As(&nextProducerMutex);
        if (producerResult != S_OK) return;
        ComPtr<IDXGIResource1> resource;
        producerResult = nextSharedProducer.As(&resource);
        if (producerResult != S_OK) return;
        producerResult = resource->CreateSharedHandle(nullptr,
            DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE,
            nullptr, &nextHandle);
      });
    } catch (...) {}
    if (!callbackRan) {
      if (nextHandle) CloseHandle(nextHandle);
      return E_ABORT;
    }
    if (producerResult != S_OK) {
      if (nextHandle) CloseHandle(nextHandle);
      return producerResult;
    }

    ComPtr<ID3D11Texture2D> nextSharedConsumer;
    ComPtr<IDXGIKeyedMutex> nextConsumerMutex;
    ComPtr<ID3D11Texture2D> nextPrivate;
    ComPtr<ID3D11Query> nextEvent;
    HRESULT result = consumerDevice1->OpenSharedResource1(
        nextHandle, IID_PPV_ARGS(&nextSharedConsumer));
    CloseHandle(nextHandle);
    nextHandle = nullptr;
    if (result == S_OK && nextSharedConsumer)
      result = nextSharedConsumer.As(&nextConsumerMutex);
    if (result == S_OK && nextConsumerMutex)
      result = consumer->CreateTexture2D(&privateDescription, nullptr, &nextPrivate);
    if (result == S_OK && nextPrivate) {
      const D3D11_QUERY_DESC queryDescription{D3D11_QUERY_EVENT, 0};
      result = consumer->CreateQuery(&queryDescription, &nextEvent);
    }
    if (result != S_OK || !nextSharedConsumer || !nextConsumerMutex ||
        !nextPrivate || !nextEvent) {
      if (nextHandle) CloseHandle(nextHandle);
      return result == S_OK ? E_FAIL : result;
    }

    // No handoff is pending here, so replacing a resized/reformatted slot
    // cannot strand a producer mutex. Old COM resources are released only
    // after the prior transfer has completed.
    clearResources();
    sharedProducer = std::move(nextSharedProducer);
    sharedConsumer = std::move(nextSharedConsumer);
    privateTexture = std::move(nextPrivate);
    producerMutex = std::move(nextProducerMutex);
    consumerMutex = std::move(nextConsumerMutex);
    consumerEvent = std::move(nextEvent);
    sourceWidth = sourceDescription.Width;
    sourceHeight = sourceDescription.Height;
    sourceFamily = choice.family;
    return S_OK;
  }

  HRESULT publish(ID3D11Texture2D*& output) noexcept {
    if (!privateTexture) return E_FAIL;
    output = privateTexture.Get();
    return S_OK;
  }

  HRESULT producerCopy(ID3D11Texture2D* source, DWORD timeoutMs,
                       TransferWallTimes* times, ProducerGpuTiming* producerTiming) noexcept {
    if (!source || !hasResources()) return E_INVALIDARG;
    if (fenced()) return producerCopyFenced(source, times, producerTiming);
    HRESULT mutexResult = E_FAIL;
    bool callbackRan = false;
    bool callbackCompleted = false;
    try {
      SubmissionWallScope dispatch(times?&times->producerDispatch:nullptr);
      (void)producerExecutor->invoke([&] {
        callbackRan = true;
        SubmissionWallScope callback(times?&times->producerCallback:nullptr);
        { SubmissionWallScope measured(times?&times->producerAcquire:nullptr);
          mutexResult = producerMutex->AcquireSync(0, timeoutMs); }
        if (mutexResult != S_OK) { callbackCompleted = true; return; }
        if (producerTiming) {
          producerTiming->poll(GetTickCount64());
          producerTiming->beginCopy(producer.Get(), producerContext.Get());
        }
        { SubmissionWallScope measured(times?&times->producerCopy:nullptr);
          producerContext->CopyResource(sharedProducer.Get(), source); }
        { SubmissionWallScope measured(times?&times->producerFlush:nullptr);
          producerContext->Flush(); }
        if (producerTiming) producerTiming->endCopy();
        const HRESULT deviceResult = producer->GetDeviceRemovedReason();
        HRESULT releaseResult;
        { SubmissionWallScope measured(times?&times->producerRelease:nullptr);
          releaseResult = producerMutex->ReleaseSync(1); }
        mutexResult = deviceResult == S_OK ? releaseResult : deviceResult;
        callbackCompleted = true;
      });
    } catch (...) {}
    if (!callbackRan) return E_ABORT;
    if (!callbackCompleted) { faulted = true; return E_FAIL; }
    if (mutexResult == WAIT_TIMEOUT) {
      pendingStage = Stage::None;
      return mutexResult;
    }
    if (mutexResult != S_OK) {
      faulted = true;
      return mutexResult;
    }
    pendingStage = Stage::ConsumerAcquire;
    return S_OK;
  }

  // The fenced producer half. The callback never waits on the CPU: the GPU
  // waits for the consumer's previous read (free >= fenceValue) before the
  // copy, and ready = fenceValue + 1 is queued after it and flushed so the
  // consumer's wait can never precede a signal still sitting in a queue.
  HRESULT producerCopyFenced(ID3D11Texture2D* source, TransferWallTimes* times,
                             ProducerGpuTiming* producerTiming) noexcept {
    HRESULT result = E_FAIL;
    bool callbackRan = false;
    bool callbackCompleted = false;
    const UINT64 next = fenceValue + 1;
    try {
      SubmissionWallScope dispatch(times?&times->producerDispatch:nullptr);
      (void)producerExecutor->invoke([&] {
        callbackRan = true;
        SubmissionWallScope callback(times?&times->producerCallback:nullptr);
        { SubmissionWallScope measured(times?&times->producerAcquire:nullptr);
          result = producerContext4->Wait(freeOnProducer.Get(), fenceValue); }
        if (result != S_OK) { callbackCompleted = true; return; }
        if (producerTiming) {
          producerTiming->poll(GetTickCount64());
          producerTiming->beginCopy(producer.Get(), producerContext.Get());
        }
        { SubmissionWallScope measured(times?&times->producerCopy:nullptr);
          producerContext->CopyResource(sharedProducer.Get(), source); }
        HRESULT signalResult;
        { SubmissionWallScope measured(times?&times->producerRelease:nullptr);
          signalResult = producerContext4->Signal(readyOnProducer.Get(), next); }
        { SubmissionWallScope measured(times?&times->producerFlush:nullptr);
          producerContext->Flush(); }
        if (producerTiming) producerTiming->endCopy();
        const HRESULT deviceResult = producer->GetDeviceRemovedReason();
        result = deviceResult == S_OK ? signalResult : deviceResult;
        callbackCompleted = true;
      });
    } catch (...) {}
    if (!callbackRan) return E_ABORT;
    if (!callbackCompleted || result != S_OK) { faulted = true; return callbackCompleted ? result : E_FAIL; }
    pendingStage = Stage::ConsumerAcquire;
    return S_OK;
  }

  // The fenced consumer half: GPU wait for ready, copy, then signal free.
  HRESULT consumeFenced(GpuWorkObserver* observer, unsigned phase,
                        TransferWallTimes* times) noexcept {
    const UINT64 next = fenceValue + 1;
    HRESULT result;
    { SubmissionWallScope measured(times?&times->consumerAcquire:nullptr);
      result = consumerContext4->Wait(readyOnConsumer.Get(), next); }
    if (result != S_OK) { faulted = true; return result; }
    if(observer) observer->beginGpuWork(phase, consumerContext.Get());
    consumerContext->CopyResource(privateTexture.Get(), sharedConsumer.Get());
    if(observer) observer->endGpuWork(phase, consumerContext.Get());
    HRESULT signalResult;
    { SubmissionWallScope measured(times?&times->consumerRelease:nullptr);
      signalResult = consumerContext4->Signal(freeOnConsumer.Get(), next); }
    { SubmissionWallScope measured(times?&times->consumerFlush:nullptr);
      consumerContext->Flush(); }
    const HRESULT deviceResult = consumer->GetDeviceRemovedReason();
    result = deviceResult == S_OK ? signalResult : deviceResult;
    if (result != S_OK) { faulted = true; return result; }
    fenceValue = next;
    pendingStage = Stage::None;
    return S_OK;
  }

  HRESULT consume(DWORD timeoutMs, GpuWorkObserver* observer = nullptr, unsigned phase = 0,
                  TransferWallTimes* times = nullptr) noexcept {
    if (!hasResources() || pendingStage != Stage::ConsumerAcquire) return E_INVALIDARG;
    if (fenced()) return consumeFenced(observer, phase, times);
    HRESULT mutexResult;
    { SubmissionWallScope measured(times?&times->consumerAcquire:nullptr);
      mutexResult = consumerMutex->AcquireSync(1, timeoutMs); }
    if (mutexResult == WAIT_TIMEOUT) {
      pendingStage = Stage::ConsumerAcquire;
      return mutexResult;
    }
    if (mutexResult != S_OK) {
      faulted = true;
      return mutexResult;
    }
    if(observer) observer->beginGpuWork(phase, consumerContext.Get());
    consumerContext->CopyResource(privateTexture.Get(), sharedConsumer.Get());
    if(observer) observer->endGpuWork(phase, consumerContext.Get());
    { SubmissionWallScope measured(times?&times->consumerFlush:nullptr);
      consumerContext->Flush(); }
    const HRESULT deviceResult = consumer->GetDeviceRemovedReason();
    HRESULT releaseResult;
    { SubmissionWallScope measured(times?&times->consumerRelease:nullptr);
      releaseResult = consumerMutex->ReleaseSync(0); }
    mutexResult = deviceResult == S_OK ? releaseResult : deviceResult;
    if (mutexResult != S_OK) {
      faulted = true;
      return mutexResult;
    }
    pendingStage = Stage::None;
    return S_OK;
  }

  HRESULT receiveInternal(ID3D11Texture2D*& output, DWORD timeoutMs,
                          GpuWorkObserver* observer = nullptr, unsigned phase = 0,
                          TransferWallTimes* times = nullptr) noexcept {
    output = nullptr;
    if (pendingStage != Stage::ConsumerAcquire) return E_UNEXPECTED;
    const HRESULT consumed = consume(timeoutMs, observer, phase, times);
    if (consumed != S_OK) return consumed;
    return publish(output);
  }

  HRESULT drainConsumer(DWORD timeoutMs) noexcept {
    if (!hasResources()) return S_OK;
    if (!consumerContext || !consumerEvent) {
      faulted = true;
      return E_FAIL;
    }
    consumerContext->End(consumerEvent.Get());
    consumerContext->Flush();
    const ULONGLONG started = GetTickCount64();
    for (;;) {
      const HRESULT deviceResult = consumer->GetDeviceRemovedReason();
      if (deviceResult != S_OK) {
        faulted = true;
        return deviceResult;
      }
      BOOL complete = FALSE;
      // Let the driver flush while polling. On RTX 5090, DONOTFLUSH stalled
      // resize drains despite End/Flush; flags=0 completed the same fixture
      // with the same deadline. This still touches only the consumer context.
      const HRESULT result = consumerContext->GetData(
          consumerEvent.Get(), &complete, sizeof(complete), 0);
      if (result == S_OK && complete) return S_OK;
      if (FAILED(result)) {
        faulted = true;
        return result;
      }
      if (GetTickCount64() - started >= timeoutMs) return timeoutResult();
      Sleep(1);
    }
  }

  HRESULT shutdown(DWORD timeoutMs) noexcept {
    if (!onOwner()) return E_ACCESSDENIED;
    if (timeoutMs == INFINITE) return E_INVALIDARG;
    if (!initialized) return S_OK;
    if (faulted) return E_FAIL;
    if (pendingStage == Stage::ConsumerAcquire) {
      ID3D11Texture2D* ignored = nullptr;
      const HRESULT received = receiveInternal(ignored, timeoutMs);
      if (received != S_OK) return received == WAIT_TIMEOUT ? E_PENDING : received;
    }
    const HRESULT drained = drainConsumer(timeoutMs);
    if (drained != S_OK) return drained;
    clearResources();
    pendingStage = Stage::None;
    consumerDevice1.Reset();
    consumerContext4.Reset();
    producerContext4.Reset();
    consumerDevice5.Reset();
    producerDevice5.Reset();
    consumerContext.Reset();
    producerContext.Reset();
    consumer.Reset();
    producer.Reset();
    producerExecutor = nullptr;
    initialized = false;
    return S_OK;
  }

  std::thread::id owner;
  ComPtr<ID3D11Device> producer;
  ComPtr<ID3D11Device> consumer;
  ComPtr<ID3D11DeviceContext> producerContext;
  ComPtr<ID3D11DeviceContext> consumerContext;
  ComPtr<ID3D11Device1> consumerDevice1;
  ImmediateExecutor* producerExecutor = nullptr;
  bool initialized = false;
  bool faulted = false;
  Stage pendingStage = Stage::None;
  ComPtr<ID3D11Texture2D> sharedProducer;
  ComPtr<ID3D11Texture2D> sharedConsumer;
  ComPtr<ID3D11Texture2D> privateTexture;
  ComPtr<IDXGIKeyedMutex> producerMutex;
  ComPtr<IDXGIKeyedMutex> consumerMutex;
  ComPtr<ID3D11Query> consumerEvent;
  // Fenced handoff (see createFencedResources). ready: producer signals,
  // consumer waits. free: consumer signals, producer waits. fenceValue is the
  // last value both reached.
  ComPtr<ID3D11Device5> producerDevice5;
  ComPtr<ID3D11Device5> consumerDevice5;
  ComPtr<ID3D11DeviceContext4> producerContext4;
  ComPtr<ID3D11DeviceContext4> consumerContext4;
  ComPtr<ID3D11Fence> readyOnProducer;
  ComPtr<ID3D11Fence> readyOnConsumer;
  ComPtr<ID3D11Fence> freeOnConsumer;
  ComPtr<ID3D11Fence> freeOnProducer;
  UINT64 fenceValue = 0;
  bool fencesRefused = false;
  bool syncReported = false;
  UINT sourceWidth = 0, sourceHeight = 0;
  PixelFamily sourceFamily = PixelFamily::RGBA8;
};

SharedTextureTransfer::SharedTextureTransfer() : impl_(new Impl()) {}

SharedTextureTransfer::~SharedTextureTransfer() {
  if (impl_ && impl_->initialized) std::terminate();
}

HRESULT SharedTextureTransfer::initialize(ID3D11Device* producer,
                                          ID3D11Device* consumer,
                                          ImmediateExecutor* producerExecutor) noexcept {
  if (!impl_) return E_FAIL;
  try {
    return impl_->validateInitialize(producer, consumer, producerExecutor);
  } catch (...) {
    return E_FAIL;
  }
}

HRESULT SharedTextureTransfer::copy(ID3D11Texture2D* source,
                                    ID3D11Texture2D*& output,
                                    DWORD timeoutMs, GpuWorkObserver* observer,
                                    unsigned phase) noexcept {
  output = nullptr;
  const HRESULT queued=enqueue(source,timeoutMs);
  return queued==S_OK ? receive(output,timeoutMs,observer,phase) : queued;
}

HRESULT SharedTextureTransfer::enqueue(ID3D11Texture2D* source, DWORD timeoutMs,
                                       TransferWallTimes* times,
                                       ProducerGpuTiming* producerTiming) noexcept {
  if (!impl_ || !impl_->onOwner()) return E_ACCESSDENIED;
  if (timeoutMs == INFINITE) return E_INVALIDARG;
  if (!impl_->initialized || impl_->faulted) return E_FAIL;
  try {
    if (impl_->pendingStage != Impl::Stage::None) return E_PENDING;
    if (!source) return E_INVALIDARG;
    D3D11_TEXTURE2D_DESC sourceDescription{};
    source->GetDesc(&sourceDescription);
    FormatChoice choice{};
    if (!validSource(sourceDescription) || !formatChoice(sourceDescription.Format, choice))
      return E_INVALIDARG;
    ComPtr<ID3D11Device> sourceDevice;
    source->GetDevice(&sourceDevice);
    if (!sameDevice(sourceDevice.Get(), impl_->producer.Get())) return E_INVALIDARG;
    const HRESULT resources = impl_->createResources(sourceDescription, choice, timeoutMs);
    if (resources != S_OK) return resources;
    return impl_->producerCopy(source, timeoutMs, times, producerTiming);
  } catch (...) {
    impl_->faulted = true;
    return E_FAIL;
  }
}

HRESULT SharedTextureTransfer::receive(ID3D11Texture2D*& output,
                                       DWORD timeoutMs, GpuWorkObserver* observer,
                                       unsigned phase, TransferWallTimes* times) noexcept {
  output = nullptr;
  if (!impl_ || !impl_->onOwner()) return E_ACCESSDENIED;
  if (timeoutMs == INFINITE) return E_INVALIDARG;
  if (!impl_->initialized || impl_->faulted) return E_FAIL;
  try {
    return impl_->receiveInternal(output, timeoutMs, observer, phase, times);
  } catch (...) {
    impl_->faulted = true;
    return E_FAIL;
  }
}

HRESULT SharedTextureTransfer::shutdown(DWORD timeoutMs) noexcept {
  if (!impl_) return E_FAIL;
  try {
    return impl_->shutdown(timeoutMs);
  } catch (...) {
    impl_->faulted = true;
    return E_FAIL;
  }
}

bool SharedTextureTransfer::ready() const noexcept {
  return impl_ && impl_->onOwner() && impl_->initialized && !impl_->faulted &&
      impl_->pendingStage == Impl::Stage::None;
}

bool SharedTextureTransfer::pending() const noexcept {
  return impl_ && impl_->onOwner() && impl_->pendingStage != Impl::Stage::None;
}

bool SharedTextureTransfer::faulted() const noexcept {
  return impl_ && impl_->onOwner() && impl_->faulted;
}

}  // namespace edvr::openxr
