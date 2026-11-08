// nvml_poll_demo.c
// Companion example for: topnotchnote.com/gpu/gpu_management.html
//
// Read GPU telemetry the way monitoring agents do, through NVML (the library
// nvidia-smi is built on), and time it:
//   1. nvmlInit cost
//   2. per-call latency of common queries (temperature, power, utilization,
//      memory, SM clock, PCIe link), 1,000 calls each
//   3. one full "sample" of all six, which is what a poller does each tick
// Compare with launching `nvidia-smi --query-gpu=...` once per sample, which
// starts a process and initializes NVML every time (time it with the shell
// loop on the page).
// The AMD counterpart is the AMD SMI C library (amdsmi_init,
// amdsmi_get_temp_metric, amdsmi_get_power_info, amdsmi_get_gpu_activity, ...).
//
// Compile: gcc -O2 -o nvml_poll nvml_poll_demo.c -lnvidia-ml

#include <nvml.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NVML_CHECK(call)                                                       \
  do {                                                                         \
    nvmlReturn_t r_ = (call);                                                  \
    if (r_ != NVML_SUCCESS) {                                                  \
      fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, nvmlErrorString(r_)); \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

static double now_us(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1e6 + t.tv_nsec / 1e3;
}

static nvmlDevice_t dev;
static unsigned int u;
static nvmlUtilization_t util;
static nvmlMemory_t mem;

static nvmlReturn_t q_temp(void)  { return nvmlDeviceGetTemperature(dev, NVML_TEMPERATURE_GPU, &u); }
static nvmlReturn_t q_power(void) { return nvmlDeviceGetPowerUsage(dev, &u); }
static nvmlReturn_t q_util(void)  { return nvmlDeviceGetUtilizationRates(dev, &util); }
static nvmlReturn_t q_mem(void)   { return nvmlDeviceGetMemoryInfo(dev, &mem); }
static nvmlReturn_t q_clock(void) { return nvmlDeviceGetClockInfo(dev, NVML_CLOCK_SM, &u); }
static nvmlReturn_t q_pcie(void)  { return nvmlDeviceGetCurrPcieLinkWidth(dev, &u); }

int main(void) {
  double t0 = now_us();
  NVML_CHECK(nvmlInit_v2());
  double t_init = now_us() - t0;
  NVML_CHECK(nvmlDeviceGetHandleByIndex_v2(0, &dev));
  char name[NVML_DEVICE_NAME_BUFFER_SIZE], drv[NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE];
  NVML_CHECK(nvmlDeviceGetName(dev, name, sizeof name));
  NVML_CHECK(nvmlSystemGetDriverVersion(drv, sizeof drv));
  printf("%s, driver %s\n", name, drv);
  printf("1. nvmlInit: %.1f ms\n\n2. per call (1,000 calls each):\n", t_init / 1e3);

  struct { const char* name; nvmlReturn_t (*fn)(void); } qs[] = {
      {"temperature", q_temp}, {"power", q_power}, {"utilization", q_util},
      {"memory used", q_mem},  {"SM clock", q_clock}, {"PCIe width", q_pcie}};
  const int N = 1000;
  for (int i = 0; i < 6; ++i) {
    nvmlReturn_t r = qs[i].fn();
    if (r != NVML_SUCCESS) { printf("   %-12s %s\n", qs[i].name, nvmlErrorString(r)); continue; }
    t0 = now_us();
    for (int k = 0; k < N; ++k) qs[i].fn();
    printf("   %-12s %8.1f us\n", qs[i].name, (now_us() - t0) / N);
  }

  t0 = now_us();
  for (int k = 0; k < N; ++k)
    for (int i = 0; i < 6; ++i) qs[i].fn();
  printf("\n3. one sample of all six: %.2f ms\n", (now_us() - t0) / N / 1e3);
  unsigned int temp_c, power_mw;
  NVML_CHECK(nvmlDeviceGetTemperature(dev, NVML_TEMPERATURE_GPU, &temp_c));
  q_util();
  q_mem();
  printf("   now: %u C, GPU busy %u%%, %.0f of %.0f MiB used", temp_c, util.gpu,
         mem.used / 1048576.0, mem.total / 1048576.0);
  if (nvmlDeviceGetPowerUsage(dev, &power_mw) == NVML_SUCCESS) printf(", %.1f W", power_mw / 1e3);
  printf("\n");
  NVML_CHECK(nvmlShutdown());
  return 0;
}
