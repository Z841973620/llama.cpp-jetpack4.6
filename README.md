### 为 JetPack4.6 编译 llama.cpp，支持 Jetson Xavier/TX2/TX1/Nano

![IMG](./IMG.png)

基于 llama.cpp b5556，删除了 bf16 支持以在 cuda10.2 环境编译

需从源码构建 gcc-8.5，默认自带的 gcc-7 缺少功能 ```vld1q_s8_x4```

编译 llama.cpp 使用 cmake-3.22.1

```
# compile gcc-8.5 from source
curl -fkLO https://bigsearcher.com/mirrors/gcc/releases/gcc-8.5.0/gcc-8.5.0.tar.gz
tar -zvxf gcc-8.5.0.tar.gz --directory=/usr/local/ && cd /usr/local/gcc-8.5.0/
./contrib/download_prerequisites
mkdir build && cd build && ../configure -enable-checking=release -enable-languages=c,c++
make && make install
```
```
# compile llama.cpp for sm_53, sm_62 and sm_72
git clone https://github.com/Z841973620/llama.cpp-tegra.git && cd llama.cpp-tegra/llama.cpp
cmake -B build \
	-DLLAMA_CURL=ON -DBUILD_SHARED_LIBS=OFF \
	-DGGML_CPU_ARM_ARCH=native -DGGML_NATIVE=OFF \
	-DGGML_CUDA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON -DCMAKE_CUDA_ARCHITECTURES="53;62;72"
cmake --build build --config Release -j --target llama-server llama-cli llama-bench llama-quantize
```
