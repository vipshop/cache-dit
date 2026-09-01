---
url: https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/graphics-interop.html
---

# 4.19. CUDA Interoperability with APIs

Directly accessing GPU data from APIs in CUDA allows to read and write the data with CUDA kernels and thereby offering CUDA features while consuming them from other APIs. There are two main concepts: the direct approach, [Graphics Interoperability](#graphics-interoperability) with openGL and Direct3D[9-11] which enables to map the resources from the OpenGL and the Direct3D to the CUDA address space; and the more flexible [External resource interoperability](#external-resource-interoperability), where memory and synchronization objects can be accessed by importing and exporting through OS-level handles. This is supported for the following APIs, Direct3D[11-12], Vulkan and the NVIDIA Software Communication Interface Interoperability.

## 4.19.1. Graphics Interoperability

Before accessing a Direct3D or an openGL resource, for example a VBO (vertex buffer object) with CUDA, it must be registered and mapped. The registering with the according CUDA functions, see examples below, returns a CUDA graphics resource of type `struct cudaGraphicsResource`, which holds a CUDA device pointer or array. To access the device data in a kernel, the resource must be mapped. While the resource is registered, it can be mapped and unmapped as many times as necessary. A mapped resource is accessed by kernels using the device memory address returned by `cudaGraphicsResourceGetMappedPointer()` for buffers and `cudaGraphicsSubResourceGetMappedArray()` for CUDA arrays. Once the resource is no longer needed by CUDA, it can be unregistered. These are the main steps: 1\. Register the graphics buffer with CUDA 2\. Map the resource 3\. Access the device pointer or array of the mapped resource 4\. Use device pointer or array in a CUDA kernel 4\. Unmap the resource 5\. Unregister the resource

Note that, registering a resource is costly and therefore ideally only called once per resource, however for each CUDA context which intends to use the resource, it is required to register the resource separately. `cudaGraphicsResourceSetMapFlags()` can be called to specify usage hints (write-only, read-only) that the CUDA driver can use to optimize resource management. Further note, that when accessing a resource through OpenGL, Direct3D, or a different CUDA context while it is mapped, it produces undefined results.

### 4.19.1.1. OpenGL Interoperability

The OpenGL resources that can be mapped into the address space of CUDA are OpenGL buffer, texture, and renderbuffer objects. A buffer object is registered using `cudaGraphicsGLRegisterBuffer()`, in CUDA, it appears as a normal device pointer. A texture or renderbuffer object is registered using `cudaGraphicsGLRegisterImage()`, in CUDA, it appears as a CUDA array.

If a texture or render buffer object has been registered with the `cudaGraphicsRegisterFlagsSurfaceLoadStore` flag, it can be written to. `cudaGraphicsGLRegisterImage()` supports all texture formats with 1, 2, or 4 components and an internal type of float (for example, `GL_RGBA_FLOAT32`), normalized integer (for example, `GL_RGBA8, GL_INTENSITY16`), and unnormalized integer (for example, `GL_RGBA8UI`).

**Example:simpleGL interoperability**

The following code sample uses a kernel to dynamically modify a 2D `width` x `height` grid of vertices stored in a vertex buffer object (VBO), and goes through the following main steps:

  1. Register the VBO with CUDA

  2. Loop: Map the VBO for writing from CUDA

  3. Loop: Run CUDA kernel to modify the vertex positions

  4. Loop: Unmap the VBO

  5. Loop: Render the results using OpenGL

  6. Unregister and delete VBO


The full example, simpleGL, of this section can be found here, [NVIDIA/cuda-samples](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/5_Domain_Specific/simpleGL) .
    
    
    __global__ void simple_vbo_kernel(float4 *pos, unsigned int width, unsigned int height, float time)
    {
        unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    
        // calculate uv coordinates
        float u = x / (float)width;
        float v = y / (float)height;
        u = u * 2.0f - 1.0f;
        v = v * 2.0f - 1.0f;
    
        // calculate simple sine wave pattern
        float freq = 4.0f;
        float w = sinf(u * freq + time) * cosf(v * freq + time) * 0.5f;
    
        // write output vertex
        pos[y * width + x] = make_float4(u, w, v, 1.0f);
    }
    
    int main(int argc, char **argv)
    {
        char *ref_file = NULL;
    
        pArgc = &argc;
        pArgv = argv;
    
    #if defined(__linux__)
        setenv("DISPLAY", ":0", 0);
    #endif
    
        printf("%s starting...\n", sSDKsample);
    
        if (argc > 1) {
            if (checkCmdLineFlag(argc, (const char **)argv, "file")) {
                // In this mode, we are running non-OpenGL and doing a compare of the VBO was generated correctly
                getCmdLineArgumentString(argc, (const char **)argv, "file", (char **)&ref_file);
            }
        }
    
        printf("\n");
    
        // First initialize OpenGL context
        if (false == initGL(&argc, argv)) {
            return false;
        }
    
        // register callbacks
        glutDisplayFunc(display);
        glutKeyboardFunc(keyboard);
        glutMouseFunc(mouse);
        glutMotionFunc(motion);
        glutCloseFunc(cleanup);
    
        // Create an empty vertex buffer object (VBO)
        // 1. Register the VBO with CUDA
        createVBO(&vbo, &cuda_vbo_resource, cudaGraphicsMapFlagsWriteDiscard);
    
        // start rendering mainloop
        //  5. Render the results using OpenGL
        glutMainLoop();
    
        printf("%s completed, returned %s\n", sSDKsample, (g_TotalErrors == 0) ? "OK" : "ERROR!");
        exit(g_TotalErrors == 0 ? EXIT_SUCCESS : EXIT_FAILURE);
        
    }
    
    void createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res, unsigned int vbo_res_flags)
    {
        assert(vbo);
    
        // create buffer object
        glGenBuffers(1, vbo);
        glBindBuffer(GL_ARRAY_BUFFER, *vbo);
    
        // initialize buffer object
        unsigned int size = mesh_width * mesh_height * 4 * sizeof(float);
        glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
    
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    
        // register this buffer object with CUDA
        checkCudaErrors(cudaGraphicsGLRegisterBuffer(vbo_res, *vbo, vbo_res_flags));
    
        SDK_CHECK_ERROR_GL();
    }
    
    void display()
    {
        float4 *dptr;
        // 2. Map the VBO for writing from CUDA
        checkCudaErrors(cudaGraphicsMapResources(1, &cuda_vbo_resource, 0));
        size_t num_bytes;
        checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes, cuda_vbo_resource));
    
        // 3. Run CUDA kernel to modify the vertex positions
        //call the CUDA kernel
        dim3 block(8, 8, 1);
        dim3 grid(mesh_width / block.x, mesh_height / block.y, 1);
        simple_vbo_kernel<<<grid, block>>>(dptr, mesh_width, mesh_height, g_fAnim);
    
        //  4. Unmap the VBO    
        checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0));
    
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
        // set view matrix
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        glTranslatef(0.0, 0.0, translate_z);
        glRotatef(rotate_x, 1.0, 0.0, 0.0);
        glRotatef(rotate_y, 0.0, 1.0, 0.0);
    
        // 5. Render the updated  using OpenGL
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glVertexPointer(4, GL_FLOAT, 0, 0);
    
        glEnableClientState(GL_VERTEX_ARRAY);
        glColor3f(1.0, 0.0, 0.0);
        glDrawArrays(GL_POINTS, 0, mesh_width * mesh_height);
        glDisableClientState(GL_VERTEX_ARRAY);
    
        glutSwapBuffers();
    
        g_fAnim += 0.01f;
    
    }
    
    void deleteVBO(GLuint *vbo, struct cudaGraphicsResource *vbo_res)
    {
        // 6. Unregister and delete VBO
        checkCudaErrors(cudaGraphicsUnregisterResource(vbo_res));
    
        glBindBuffer(1, *vbo);
        glDeleteBuffers(1, vbo);
    
        *vbo = 0;
    }
    
    void cleanup()
    {
    
        if (vbo) {
            deleteVBO(&vbo, cuda_vbo_resource);
        }
    }
    

**Limitations and considerations.**

  * The OpenGL context whose resources are being shared has to be current to the host thread making any OpenGL interoperability API calls.

  * When an OpenGL texture is made bindless (say for example by requesting an image or texture handle using the `glGetTextureHandle` or `glGetImageHandle` APIs) it cannot be registered with CUDA. The application needs to register the texture for interop before requesting an image or texture handle.


### 4.19.1.2. Direct3D Interoperability

Direct3D interoperability is supported for Direct3D9, Direct3D10, and Direct3D11 but not Direct3D12, here we focus on Direct3D11, for Direct3D9 and Direct3D10 please refer to the CUDA programming guide 12.9. The Direct3D resources that may be mapped into the address space of CUDA are Direct3D buffers, textures, and surfaces. These resources are registered using `cudaGraphicsD3D11RegisterResource()`.

A CUDA context may interoperate only with Direct3D11 devices created with `DriverType` set to `D3D_DRIVER_TYPE_HARDWARE`.

**Example: 2D Texture Direct3D11 interoperability**

The following code snippets are from the simpleD3D11Texture example, [NVIDIA/cuda-samples](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/5_Domain_Specific/simpleD3D11Texture). The full example includes a lot of boiler plate DX11 code, here we focus on the CUDA side.

The CUDA kernel `cuda_kernel_texture_2d` paints a 2D texture with a moving red/green hatch pattern on a strobing blue background, it is dependent on the previous texture values. The underlying data is a 2D CUDA array, where the row offsets are defined by the pitch.
    
    
    /*
     * Paint a 2D texture with a moving red/green hatch pattern on a
     * strobing blue background.  Note that this kernel reads to and
     * writes from the texture, hence why this texture was not mapped
     * as WriteDiscard.
     */
    __global__ void cuda_kernel_texture_2d(unsigned char *surface, int width,
                                           int height, size_t pitch, float t) {
      int x = blockIdx.x * blockDim.x + threadIdx.x;
      int y = blockIdx.y * blockDim.y + threadIdx.y;
      float *pixel;
    
      // in the case where, due to quantization into grids, we have
      // more threads than pixels, skip the threads which don't
      // correspond to valid pixels
      if (x >= width || y >= height) return;
    
      // get a pointer to the pixel at (x,y)
      pixel = (float *)(surface + y * pitch) + 4 * x;
    
      // populate it
      float value_x = 0.5f + 0.5f * cos(t + 10.0f * ((2.0f * x) / width - 1.0f));
      float value_y = 0.5f + 0.5f * cos(t + 10.0f * ((2.0f * y) / height - 1.0f));
      pixel[0] = 0.5 * pixel[0] + 0.5 * pow(value_x, 3.0f);  // red
      pixel[1] = 0.5 * pixel[1] + 0.5 * pow(value_y, 3.0f);  // green
      pixel[2] = 0.5f + 0.5f * cos(t);                       // blue
      pixel[3] = 1;                                          // alpha
    }
    
    extern "C" void cuda_texture_2d(void *surface, int width, int height,
                                    size_t pitch, float t) {
      cudaError_t error = cudaSuccess;
    
      dim3 Db = dim3(16, 16);  // block dimensions are fixed to be 256 threads
      dim3 Dg = dim3((width + Db.x - 1) / Db.x, (height + Db.y - 1) / Db.y);
    
      cuda_kernel_texture_2d<<<Dg, Db>>>((unsigned char *)surface, width, height,
                                         pitch, t);
    
      error = cudaGetLastError();
    
      if (error != cudaSuccess) {
        printf("cuda_kernel_texture_2d() failed to launch error = %d\n", error);
      }
    }
    

To keep the pointers and data buffers belonging together the following data structure is used:
    
    
    // Data structure for 2D texture shared between DX11 and CUDA
    struct {
      ID3D11Texture2D *pTexture;
      ID3D11ShaderResourceView *pSRView;
      cudaGraphicsResource *cudaResource;
      void *cudaLinearMemory;
      size_t pitch;
      int width;
      int height;
      int offsetInShader;
    } g_texture_2d;
    

After the initialization of the Direct3D device and the textures, the resources are registered with CUDA once. To match the Direct3D pixel format, the CUDA array is allocated with the same width and height, and a pitch matching the Direct3D texture row pitch.
    
    
        // register the Direct3D resources that are used in the CUDA kernel
        // we'll read to and write from g_texture_2d, so don't set any special map flags for it
        cudaGraphicsD3D11RegisterResource(&g_texture_2d.cudaResource,
                                          g_texture_2d.pTexture,
                                          cudaGraphicsRegisterFlagsNone);
        getLastCudaError("cudaGraphicsD3D11RegisterResource (g_texture_2d) failed");
        // CUDA cannot write into the texture directly : the texture is seen as a
        // cudaArray and can only be mapped as a texture
        // Create a buffer so that CUDA can write into it
        // the pixel fmt is DXGI_FORMAT_R32G32B32A32_FLOAT
        cudaMallocPitch(&g_texture_2d.cudaLinearMemory, &g_texture_2d.pitch,
                        g_texture_2d.width * sizeof(float) * 4,
                        g_texture_2d.height);
        getLastCudaError("cudaMallocPitch (g_texture_2d) failed");
        cudaMemset(g_texture_2d.cudaLinearMemory, 1,
                   g_texture_2d.pitch * g_texture_2d.height);
    

In the rendering loop, the resources are mapped, the CUDA kernel is launched to update the texture data, and then the resources are unmapped. After this step the Direct3D device is used to draw the updated textures on the screen.
    
    
        cudaStream_t stream = 0;
        const int nbResources = 3;
        cudaGraphicsResource *ppResources[nbResources] = {
            g_texture_2d.cudaResource, g_texture_3d.cudaResource,
            g_texture_cube.cudaResource,
        };
        cudaGraphicsMapResources(nbResources, ppResources, stream);
        getLastCudaError("cudaGraphicsMapResources(3) failed");
    
        // run kernels which will populate the contents of those textures
        RunKernels();
    
        // unmap the resources
        cudaGraphicsUnmapResources(nbResources, ppResources, stream);
        getLastCudaError("cudaGraphicsUnmapResources(3) failed");
    

Finally, once the resources are no longer needed in CUDA, they are unregistered and the device array freed.
    
    
      // unregister the Cuda resources
      cudaGraphicsUnregisterResource(g_texture_2d.cudaResource);
      getLastCudaError("cudaGraphicsUnregisterResource (g_texture_2d) failed");
      cudaFree(g_texture_2d.cudaLinearMemory);
      getLastCudaError("cudaFree (g_texture_2d) failed");
    

### 4.19.1.3. Interoperability in a Scalable Link Interface (SLI) configuration

In a system with multiple GPUs, all CUDA-enabled GPUs are accessible via the CUDA driver and runtime as separate devices. This is different when the system is in SLI mode. SLI is a hardware configured multi-GPU configuration that offers increased rendering performance by dividing the workload across multiple GPUs. Implicit SLI mode, where the driver makes assumption is no longer supported, however explicit SLI is still supported. Explicit SLI means applications know and manage the SLI state through APIs (e.g. Vulkan, DirectX, GL) for all devices in the SLI group.

There are special considerations when the system is in SLI mode:

  * An allocation in one CUDA device on one GPU will consume memory on other GPUs that are part of the SLI configuration of the Direct3D or OpenGL device. Because of this, allocations may fail earlier than otherwise expected.

  * An applications should create multiple CUDA contexts, one for each GPU in the SLI configuration. While this is not a strict requirement, it avoids unnecessary data transfers between devices. The application can use the `cudaD3D[9|10|11]GetDevices()` for Direct3D and `cudaGLGetDevices()` for OpenGL set of calls to identify the CUDA device handles for the devices that are performing the rendering in the current and next frame. Given this information the application will typically choose the appropriate device and map Direct3D or OpenGL resources to the CUDA device returned by `cudaD3D[9|10|11]GetDevices()` or `cudaGLGetDevices()` when the `deviceList` parameter is set to `cudaD3D[9|10|11]DeviceListCurrentFrame` or `cudaGLDeviceListCurrentFrame`.

  * Resource returned from `cudaGraphicsD3D[9|10|11]RegisterResource` and `cudaGraphicsGLRegister[Buffer|Image]` must be only used on the device where the registration happened. Therefore, in SLI configurations when data for different frames is computed on different CUDA devices it is necessary to register the resources for each separately.


## 4.19.2. External resource interoperability

External resource interoperability allows CUDA to import certain resources that are explicitly exported by APIs. These objects are typically exported using handles native to the Operating System, like file descriptors on Linux or NT handles on Windows. This allows to efficiently share the resource between other APIs and CUDA without the need to copy or duplicate in between. It is supported for the following APIs, Direct3D[11-12], Vulkan and the NVIDIA Software Communication Interface Interoperability. There are two types of resources that can be imported:

  * **Memory objects**
    

can be imported into CUDA using `cudaImportExternalMemory()`. An imported memory object can then be accessed from within kernels using device pointers mapped onto the memory object with `cudaExternalMemoryGetMappedBuffer()` or CUDA mipmapped arrays mapped with `cudaExternalMemoryGetMappedMipmappedArray()`. Depending on the type of memory object, it may be possible for more than one mapping to be setup on a single memory object. The mappings must match the mappings setup of the exporting API. Any mismatched mappings result in undefined behavior. Imported memory objects must be freed using `cudaDestroyExternalMemory()`. Freeing a memory object does not free any mappings to that object. Therefore, any device pointers mapped onto that object must be explicitly freed using `cudaFree()` and any CUDA mipmapped arrays mapped onto that object must be explicitly freed using `cudaFreeMipmappedArray()`. It is illegal to access mappings to an object after it has been destroyed.

  * **Synchronization objects**
    

can be imported into CUDA using `cudaImportExternalSemaphore()`. An imported synchronization object can then be signaled using `cudaSignalExternalSemaphoresAsync()` and waited on using `cudaWaitExternalSemaphoresAsync()`. It is illegal to issue a wait before the corresponding signal has been issued. Also, depending on the type of the imported synchronization object, there may be additional constraints imposed on how they can be signaled and waited on, as described in subsequent sections. Imported semaphore objects must be freed using `cudaDestroyExternalSemaphore()`. All outstanding signals and waits must have completed before the semaphore object is destroyed.


### 4.19.2.1. Vulkan interoperability

Coupled execution of Vulkan graphics and compute workloads on the same hardware can maximize GPU utilization and avoid unnecessary copies. Note, this is not a Vulkan guide, we only focus on the interoperability with CUDA, for a Vulkan guide please refer to <https://www.vulkan.org/learn#vulkan-tutorials>.

The main steps to get a Vulkan-CUDA interoperability working involve:

  1. Initialize Vulkan, create and export the external buffers and/or synchronization objects

  2. Set the CUDA device where Vulkan is running with the matching devices UUIDs

  3. Get the memory and/or synchronization handle

  4. Import the memory and/or synchronization object in CUDA using these handles

  5. Map the device pointer or mipmapped array onto the memory object

  6. Use the imported memory objects in CUDA and Vulkan interchangeably by defining an order of execution through signaling and waiting on the synchronization objects.


In this section the steps above are explained with the help of the _simpleVulkan_ example, [NVIDIA/cuda-samples](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/5_Domain_Specific/simpleVulkan). We walk through the example step by step, focusing on the parts needed for the CUDA interoperability. Some variation are explained with standalone snippets.

Note

The code example used in this section, uses the direct memory allocation and resource creation. Which is not state of the art due to several reasons, including the limitation to the number of instances that can be created. However, to understand the interoperability, one needs to know the underlying Vulkan code and the specific flags. For a more state of the art example, using the [VulkanMemoryAllocator](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator) please refer to the _sample_cuda_interop_ in the [NVProSamples](https://github.com/nvpro-samples) repository.

The following data structure is used throughout the example:
    
    
    class VulkanCudaSineWave : public VulkanBaseApp {
      typedef struct UniformBufferObject_st {
        mat4x4 modelViewProj;
      } UniformBufferObject;
    
      VkBuffer m_heightBuffer, m_xyBuffer, m_indexBuffer;
      VkDeviceMemory m_heightMemory, m_xyMemory, m_indexMemory;
      UniformBufferObject m_ubo;
      VkSemaphore m_vkWaitSemaphore, m_vkSignalSemaphore;
      SineWaveSimulation m_sim;
      cudaStream_t m_stream;
      cudaExternalSemaphore_t m_cudaWaitSemaphore, m_cudaSignalSemaphore, m_cudaTimelineSemaphore;
      cudaExternalMemory_t m_cudaVertMem;
      float *m_cudaHeightMap;
      // ...
    

#### 4.19.2.1.1. Setting up a Vulkan device

In order to export memory objects, a Vulkan instance must be created with the `VK_KHR_external_memory_capabilities` extension enabled and the device with `VK_KHR_external_memory`. In addition to the platform specific handle types must be enabled, for Windows `VK_KHR_external_memory_win32` and for UNIX based systems `VK_KHR_external_memory_fd`.

Similarly for exporting synchronization objects, on the device level `VK_KHR_external_semaphore_capabilities` and `VK_KHR_external_semaphore` on the instance level need to be enabled. As well as the platform specific extensions for the handles, that is `VK_KHR_external_semaphore_win32` for Windows and `VK_KHR_external_semaphore_fd` for Unix based systems.

In the _simpleVulkan_ example these extensions are enabled with the following enums.
    
    
      std::vector<const char *> getRequiredExtensions() const {
        std::vector<const char *> extensions;
        extensions.push_back(VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);
        extensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME);
        return extensions;
      }
    
      std::vector<const char *> getRequiredDeviceExtensions() const {
        std::vector<const char *> extensions;
        extensions.push_back(VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);
        extensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME);
        extensions.push_back(VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME);
    #ifdef _WIN64
        extensions.push_back(VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
        extensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME);
    #else
        extensions.push_back(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
        extensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME);
    #endif /* _WIN64 */
        return extensions;
      }
    

These are then added to the Vulkan instance and device creation info, please see the _simpleVulkan_ example for details.

#### 4.19.2.1.2. Initializing CUDA with matching device UUIDs

When importing memory and synchronization objects exported by Vulkan, they must be imported and mapped on the same device as they were created on. The CUDA device that corresponds to the Vulkan physical device on which the objects were created can be determined by comparing the UUID of a CUDA device with that of the Vulkan physical device, as shown in the following code snippet from the simpleVulkan example, where `vkDeviceUUID` is the member of the Vulkan API structure `vkPhysicalDeviceIDProperties.deviceUUID` and defines the physical devices id of the current Vulkan instance.
    
    
    // from the CUDA example `simpleVulkan`
    int SineWaveSimulation::initCuda(uint8_t *vkDeviceUUID, size_t UUID_SIZE) {
      int current_device = 0;
      int device_count = 0;
      int devices_prohibited = 0;
    
      cudaDeviceProp deviceProp;
      checkCudaErrors(cudaGetDeviceCount(&device_count));
    
      if (device_count == 0) {
        fprintf(stderr, "CUDA error: no devices supporting CUDA.\n");
        exit(EXIT_FAILURE);
      }
    
      // Find the GPU which is selected by Vulkan
      while (current_device < device_count) {
        cudaGetDeviceProperties(&deviceProp, current_device);
    
        if ((deviceProp.computeMode != cudaComputeModeProhibited)) {
          // Compare the cuda device UUID with vulkan UUID
          int ret = memcmp((void *)&deviceProp.uuid, vkDeviceUUID, UUID_SIZE);
          if (ret == 0) {
            checkCudaErrors(cudaSetDevice(current_device));
            checkCudaErrors(cudaGetDeviceProperties(&deviceProp, current_device));
            printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n",
                   current_device, deviceProp.name, deviceProp.major,
                   deviceProp.minor);
    
            return current_device;
          }
    
        } else {
          devices_prohibited++;
        }
    
        current_device++;
      }
    
      if (devices_prohibited == device_count) {
        fprintf(stderr,
                "CUDA error:"
                " No Vulkan-CUDA Interop capable GPU found.\n");
        exit(EXIT_FAILURE);
      }
    
      return -1;
    }
    

Note that the Vulkan physical device should not be part of a device group that contains more than one Vulkan physical device. That is, the device group as returned by `vkEnumeratePhysicalDeviceGroups` that contains the given Vulkan physical device must have a physical device count of 1.

#### 4.19.2.1.3. Exporting Vulkan memory objects

In order to export a Vulkan memory object, a buffer with the according export flags must be created. Note that the enums for the handle types are platform specific.
    
    
    void VulkanBaseApp::createExternalBuffer(
        VkDeviceSize size, VkBufferUsageFlags usage,
        VkMemoryPropertyFlags properties,
        VkExternalMemoryHandleTypeFlagsKHR extMemHandleType, VkBuffer &buffer,
        VkDeviceMemory &bufferMemory) {
      VkBufferCreateInfo bufferInfo = {};
      bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
      bufferInfo.size = size;
      bufferInfo.usage = usage;
      bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    
      VkExternalMemoryBufferCreateInfo externalMemoryBufferInfo = {};
      externalMemoryBufferInfo.sType =
          VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
      externalMemoryBufferInfo.handleTypes = extMemHandleType;
      bufferInfo.pNext = &externalMemoryBufferInfo;
    
      if (vkCreateBuffer(m_device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to create buffer!");
      }
    
      VkMemoryRequirements memRequirements;
      vkGetBufferMemoryRequirements(m_device, buffer, &memRequirements);
    
    #ifdef _WIN64
      WindowsSecurityAttributes winSecurityAttributes;
    
      VkExportMemoryWin32HandleInfoKHR vulkanExportMemoryWin32HandleInfoKHR = {};
      vulkanExportMemoryWin32HandleInfoKHR.sType =
          VK_STRUCTURE_TYPE_EXPORT_MEMORY_WIN32_HANDLE_INFO_KHR;
      vulkanExportMemoryWin32HandleInfoKHR.pNext = NULL;
      vulkanExportMemoryWin32HandleInfoKHR.pAttributes = &winSecurityAttributes;
      vulkanExportMemoryWin32HandleInfoKHR.dwAccess =
          DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE;
      vulkanExportMemoryWin32HandleInfoKHR.name = (LPCWSTR)NULL;
    #endif /* _WIN64 */
      VkExportMemoryAllocateInfoKHR vulkanExportMemoryAllocateInfoKHR = {};
      vulkanExportMemoryAllocateInfoKHR.sType =
          VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO_KHR;
    #ifdef _WIN64
      vulkanExportMemoryAllocateInfoKHR.pNext =
          extMemHandleType & VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT_KHR
              ? &vulkanExportMemoryWin32HandleInfoKHR
              : NULL;
      vulkanExportMemoryAllocateInfoKHR.handleTypes = extMemHandleType;
    #else
      vulkanExportMemoryAllocateInfoKHR.pNext = NULL;
      vulkanExportMemoryAllocateInfoKHR.handleTypes =
          VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    #endif /* _WIN64 */
      VkMemoryAllocateInfo allocInfo = {};
      allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
      allocInfo.pNext = &vulkanExportMemoryAllocateInfoKHR;
      allocInfo.allocationSize = memRequirements.size;
      allocInfo.memoryTypeIndex = findMemoryType(
          m_physicalDevice, memRequirements.memoryTypeBits, properties);
    
      if (vkAllocateMemory(m_device, &allocInfo, nullptr, &bufferMemory) !=
          VK_SUCCESS) {
        throw std::runtime_error("failed to allocate external buffer memory!");
      }
    
      vkBindBufferMemory(m_device, buffer, bufferMemory, 0);
    }
    

#### 4.19.2.1.4. Exporting Vulkan synchronization objects

Vulkan API calls which are executed on the GPU are asynchronous. To define an order of execution there are semaphores and fences available in Vulkan which can be shared with CUDA. Similar to the memory objects, semaphores can be exported by Vulkan, they need to be created with the export flags depending on the type of semaphore. There are binary and timeline semaphores. Binary semaphores only have a 1 bit counter, either signaled or not. Timeline semaphores have a 64 bit counter, which can be used to define an order of execution with the same semaphore. In the _simpleVulkan_ example there are code paths for both timeline and binary semaphores.
    
    
    void VulkanBaseApp::createExternalSemaphore(
        VkSemaphore &semaphore, VkExternalSemaphoreHandleTypeFlagBits handleType) {
      VkSemaphoreCreateInfo semaphoreInfo = {};
      semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
      VkExportSemaphoreCreateInfoKHR exportSemaphoreCreateInfo = {};
      exportSemaphoreCreateInfo.sType =
          VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO_KHR;
    
    #ifdef _VK_TIMELINE_SEMAPHORE
      VkSemaphoreTypeCreateInfo timelineCreateInfo;
      timelineCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
      timelineCreateInfo.pNext = NULL;
      timelineCreateInfo.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
      timelineCreateInfo.initialValue = 0;
      exportSemaphoreCreateInfo.pNext = &timelineCreateInfo;
    #else
      exportSemaphoreCreateInfo.pNext = NULL;
    #endif /* _VK_TIMELINE_SEMAPHORE */
      exportSemaphoreCreateInfo.handleTypes = handleType;
      semaphoreInfo.pNext = &exportSemaphoreCreateInfo;
    
      if (vkCreateSemaphore(m_device, &semaphoreInfo, nullptr, &semaphore) !=
          VK_SUCCESS) {
        throw std::runtime_error(
            "failed to create synchronization objects for a CUDA-Vulkan!");
      }
    }
    

#### 4.19.2.1.5. Importing memory objects

Both dedicated and non-dedicated memory objects exported by Vulkan can be imported into CUDA. When importing a Vulkan dedicated memory object, the flag `cudaExternalMemoryDedicated` must be set.

In Windows, a Vulkan memory object exported using `VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT` can be imported into CUDA using the NT handle associated with that object as shown below. Note that CUDA does not assume ownership of the NT handle and it is the application’s responsibility to close the handle when it is not required anymore. The NT handle holds a reference to the resource, so it must be explicitly freed before the underlying memory can be freed.

In Linux, a Vulkan memory object exported using `VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT` can be imported into CUDA using the file descriptor associated with that object as shown below. Note that CUDA assumes ownership of the file descriptor once it is imported. Using the file descriptor after a successful import results in undefined behavior.
    
    
      // from the CUDA example `simpleVulkan`
      void importCudaExternalMemory(void **cudaPtr, cudaExternalMemory_t &cudaMem,
                                    VkDeviceMemory &vkMem, VkDeviceSize size,
                                    VkExternalMemoryHandleTypeFlagBits handleType) {
        cudaExternalMemoryHandleDesc externalMemoryHandleDesc = {};
    
        if (handleType & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT) {
          externalMemoryHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
        } else if (handleType &
                   VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT) {
          externalMemoryHandleDesc.type =
              cudaExternalMemoryHandleTypeOpaqueWin32Kmt;
        } else if (handleType & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT) {
          externalMemoryHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueFd;
        } else {
          throw std::runtime_error("Unknown handle type requested!");
        }
    
        externalMemoryHandleDesc.size = size;
    
    #ifdef _WIN64
        externalMemoryHandleDesc.handle.win32.handle =
            (HANDLE)getMemHandle(vkMem, handleType);
    #else
        externalMemoryHandleDesc.handle.fd =
            (int)(uintptr_t)getMemHandle(vkMem, handleType);
    #endif
    
        checkCudaErrors(
            cudaImportExternalMemory(&cudaMem, &externalMemoryHandleDesc));
    

A Vulkan memory object exported using `VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT` can also be imported using a named handle if one exists as shown in the standalone snippet below.
    
    
    cudaExternalMemory_t importVulkanMemoryObjectFromNamedNTHandle(LPCWSTR name, unsigned long long size, bool isDedicated) {
       cudaExternalMemory_t extMem = NULL;
       cudaExternalMemoryHandleDesc desc = {};
    
       memset(&desc, 0, sizeof(desc));
    
       desc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
       desc.handle.win32.name = (void *)name;
       desc.size = size;
       if (isDedicated) {
           desc.flags |= cudaExternalMemoryDedicated;
       }
    
       cudaImportExternalMemory(&extMem, &desc);
    
       return extMem;
    }
    

#### 4.19.2.1.6. Mapping buffers onto imported memory objects

After importing a memory object, they have to be mapped before they can be used. A device pointer can be mapped onto an imported memory object as shown below. The offset and size of the mapping must match that specified when creating the mapping using the corresponding Vulkan API. All mapped device pointers must be freed using `cudaFree()`.
    
    
        // from the CUDA example `simpleVulkan`, continuation of function `importCudaExternalMemory`
        cudaExternalMemoryBufferDesc externalMemBufferDesc = {};
        externalMemBufferDesc.offset = 0;
        externalMemBufferDesc.size = size;
        externalMemBufferDesc.flags = 0;
    
        checkCudaErrors(cudaExternalMemoryGetMappedBuffer(cudaPtr, cudaMem,
                                                          &externalMemBufferDesc));
      }
    

#### 4.19.2.1.7. Mapping mipmapped arrays onto imported memory objects

A CUDA mipmapped array can be mapped onto an imported memory object as shown below. The offset, dimensions, format and number of mip levels must match that specified when creating the mapping using the corresponding Vulkan API. Additionally, if the mipmapped array is bound as a color target in Vulkan, the flag`cudaArrayColorAttachment` must be set. All mapped mipmapped arrays must be freed using `cudaFreeMipmappedArray()`. The following code standalone snippet shows how to convert Vulkan parameters into the corresponding CUDA parameters when mapping mipmapped arrays onto imported memory objects.
    
    
    cudaMipmappedArray_t mapMipmappedArrayOntoExternalMemory(cudaExternalMemory_t extMem, unsigned long long offset, cudaChannelFormatDesc *formatDesc, cudaExtent *extent, unsigned int flags, unsigned int numLevels) {
        cudaMipmappedArray_t mipmap = NULL;
        cudaExternalMemoryMipmappedArrayDesc desc = {};
    
        memset(&desc, 0, sizeof(desc));
    
        desc.offset = offset;
        desc.formatDesc = *formatDesc;
        desc.extent = *extent;
        desc.flags = flags;
        desc.numLevels = numLevels;
    
        // Note: 'mipmap' must eventually be freed using cudaFreeMipmappedArray()
        cudaExternalMemoryGetMappedMipmappedArray(&mipmap, extMem, &desc);
    
        return mipmap;
    }
    //end mapMipmappedArrayOntoExternalMemory
    
    //begin getCudaChannelFormatDescForVulkanFormat
    cudaChannelFormatDesc getCudaChannelFormatDescForVulkanFormat(VkFormat format)
    {
        cudaChannelFormatDesc d;
    
        memset(&d, 0, sizeof(d));
     
        switch (format) {
           case VK_FORMAT_R8_UINT:             d.x = 8;  d.y = 0;  d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindUnsigned; break;
           case VK_FORMAT_R8_SINT:             d.x = 8;  d.y = 0;  d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindSigned;   break;
           case VK_FORMAT_R8G8_UINT:           d.x = 8;  d.y = 8;  d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindUnsigned; break;
           case VK_FORMAT_R8G8_SINT:           d.x = 8;  d.y = 8;  d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindSigned;   break;
           case VK_FORMAT_R8G8B8A8_UINT:       d.x = 8;  d.y = 8;  d.z = 8;  d.w = 8;  d.f = cudaChannelFormatKindUnsigned; break;
           case VK_FORMAT_R8G8B8A8_SINT:       d.x = 8;  d.y = 8;  d.z = 8;  d.w = 8;  d.f = cudaChannelFormatKindSigned;   break;
           case VK_FORMAT_R16_UINT:            d.x = 16; d.y = 0;  d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindUnsigned; break;
           case VK_FORMAT_R16_SINT:            d.x = 16; d.y = 0;  d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindSigned;   break;
           case VK_FORMAT_R16G16_UINT:         d.x = 16; d.y = 16; d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindUnsigned; break;
           case VK_FORMAT_R16G16_SINT:         d.x = 16; d.y = 16; d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindSigned;   break;
           case VK_FORMAT_R16G16B16A16_UINT:   d.x = 16; d.y = 16; d.z = 16; d.w = 16; d.f = cudaChannelFormatKindUnsigned; break;
           case VK_FORMAT_R16G16B16A16_SINT:   d.x = 16; d.y = 16; d.z = 16; d.w = 16; d.f = cudaChannelFormatKindSigned;   break;
           case VK_FORMAT_R32_UINT:            d.x = 32; d.y = 0;  d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindUnsigned; break;
           case VK_FORMAT_R32_SINT:            d.x = 32; d.y = 0;  d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindSigned;   break;
           case VK_FORMAT_R32_SFLOAT:          d.x = 32; d.y = 0;  d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindFloat;    break;
           case VK_FORMAT_R32G32_UINT:         d.x = 32; d.y = 32; d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindUnsigned; break;
           case VK_FORMAT_R32G32_SINT:         d.x = 32; d.y = 32; d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindSigned;   break;
           case VK_FORMAT_R32G32_SFLOAT:       d.x = 32; d.y = 32; d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindFloat;    break;
           case VK_FORMAT_R32G32B32A32_UINT:   d.x = 32; d.y = 32; d.z = 32; d.w = 32; d.f = cudaChannelFormatKindUnsigned; break;
           case VK_FORMAT_R32G32B32A32_SINT:   d.x = 32; d.y = 32; d.z = 32; d.w = 32; d.f = cudaChannelFormatKindSigned;   break;
           case VK_FORMAT_R32G32B32A32_SFLOAT: d.x = 32; d.y = 32; d.z = 32; d.w = 32; d.f = cudaChannelFormatKindFloat;    break;
           default: assert(0);
        }
        return d;
    }
    //end getCudaChannelFormatDescForVulkanFormat
    
    //begin getCudaExtentForVulkanExtent
    cudaExtent getCudaExtentForVulkanExtent(VkExtent3D vkExt, uint32_t arrayLayers, VkImageViewType vkImageViewType) {
        cudaExtent e = { 0, 0, 0 };
    
        switch (vkImageViewType) {
            case VK_IMAGE_VIEW_TYPE_1D:         e.width = vkExt.width; e.height = 0;            e.depth = 0;           break;
            case VK_IMAGE_VIEW_TYPE_2D:         e.width = vkExt.width; e.height = vkExt.height; e.depth = 0;           break;
            case VK_IMAGE_VIEW_TYPE_3D:         e.width = vkExt.width; e.height = vkExt.height; e.depth = vkExt.depth; break;
            case VK_IMAGE_VIEW_TYPE_CUBE:       e.width = vkExt.width; e.height = vkExt.height; e.depth = arrayLayers; break;
            case VK_IMAGE_VIEW_TYPE_1D_ARRAY:   e.width = vkExt.width; e.height = 0;            e.depth = arrayLayers; break;
            case VK_IMAGE_VIEW_TYPE_2D_ARRAY:   e.width = vkExt.width; e.height = vkExt.height; e.depth = arrayLayers; break;
            case VK_IMAGE_VIEW_TYPE_CUBE_ARRAY: e.width = vkExt.width; e.height = vkExt.height; e.depth = arrayLayers; break;
            default: assert(0);
        }
    
        return e;
    }
    //end getCudaExtentForVulkanExtent
    
    //begin getCudaMipmappedArrayFlagsForVulkanImage
    unsigned int getCudaMipmappedArrayFlagsForVulkanImage(VkImageViewType vkImageViewType,
                                                          VkImageUsageFlags vkImageUsageFlags,
                                                          bool allowSurfaceLoadStore) {
        unsigned int flags = 0;
    
        switch (vkImageViewType) {
            case VK_IMAGE_VIEW_TYPE_CUBE:       flags |= cudaArrayCubemap;                    break;
            case VK_IMAGE_VIEW_TYPE_CUBE_ARRAY: flags |= cudaArrayCubemap | cudaArrayLayered; break;
            case VK_IMAGE_VIEW_TYPE_1D_ARRAY:   flags |= cudaArrayLayered;                    break;
            case VK_IMAGE_VIEW_TYPE_2D_ARRAY:   flags |= cudaArrayLayered;                    break;
            default: break;
        }
        if (vkImageUsageFlags & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) {
            flags |= cudaArrayColorAttachment;
        }
    
        if (allowSurfaceLoadStore) {
            flags |= cudaArraySurfaceLoadStore;
        }
        
        return flags;
    }
    

#### 4.19.2.1.8. Importing Synchronization Objects

A Vulkan semaphore object exported using `VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT` can be imported into CUDA using the file descriptor associated with that object as shown below. Note that CUDA assumes ownership of the file descriptor once it is imported. Using the file descriptor after a successful import results in undefined behavior.

Whereas a Vulkan semaphore object exported using `VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT` can be imported into CUDA using the NT handle associated with that object as shown below. Note that CUDA does not assume ownership of the NT handle and it is the application’s responsibility to close the handle when it is not required anymore. The NT handle holds a reference to the resource, so it must be explicitly freed before the underlying semaphore can be freed.

And, a Vulkan semaphore object exported using `VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT` can be imported into CUDA using the globally shared D3DKMT handle associated with that object as shown below. Since a globally shared D3DKMT handle does not hold a reference to the underlying semaphore it is automatically destroyed when all other references to the resource are destroyed.
    
    
      void importCudaExternalSemaphore(
          cudaExternalSemaphore_t &cudaSem, VkSemaphore &vkSem,
          VkExternalSemaphoreHandleTypeFlagBits handleType) {
        cudaExternalSemaphoreHandleDesc externalSemaphoreHandleDesc = {};
    
    #ifdef _VK_TIMELINE_SEMAPHORE
        if (handleType & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT) {
          externalSemaphoreHandleDesc.type =
              cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32;
        } else if (handleType &
                   VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT) {
          externalSemaphoreHandleDesc.type =
              cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32;
        } else if (handleType & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT) {
          externalSemaphoreHandleDesc.type =
              cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd;
        }
    #else
        if (handleType & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT) {
          externalSemaphoreHandleDesc.type =
              cudaExternalSemaphoreHandleTypeOpaqueWin32;
        } else if (handleType &
                   VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT) {
          externalSemaphoreHandleDesc.type =
              cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt;
        } else if (handleType & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT) {
          externalSemaphoreHandleDesc.type =
              cudaExternalSemaphoreHandleTypeOpaqueFd;
        }
    #endif /* _VK_TIMELINE_SEMAPHORE */
        else {
          throw std::runtime_error("Unknown handle type requested!");
        }
    
    #ifdef _WIN64
        externalSemaphoreHandleDesc.handle.win32.handle =
            (HANDLE)getSemaphoreHandle(vkSem, handleType);
    #else
        externalSemaphoreHandleDesc.handle.fd =
            (int)(uintptr_t)getSemaphoreHandle(vkSem, handleType);
    #endif
    
        externalSemaphoreHandleDesc.flags = 0;
    
        checkCudaErrors(
            cudaImportExternalSemaphore(&cudaSem, &externalSemaphoreHandleDesc));
      }
    

#### 4.19.2.1.9. Signaling/Waiting on Imported Synchronization Objects

An imported Vulkan semaphore can be signaled and waited on as shown below. Signaling a semaphore sets it to the signaled state and in the case of timeline semaphores it sets the counter to the value specified in the signal call. The corresponding wait that waits on this signal must be issued in Vulkan. Additionally, in the case of a binary semaphore, the wait that waits on this signal must be issued after this signal has been issued.

Waiting on a semaphore waits until it reaches the signaled state or the assigned wait value. A signaled binary semaphore then resets it back to the unsignaled state. The corresponding signal that this wait is waiting on must be issued in Vulkan. Additionally, in the case of a binary semaphore, the signal must be issued before this wait can be issued.

In the following code extract from the _simpleVulkan_ example the simulation step / the CUDA kernel is only called once the semaphore around the vertex buffers is signaled by Vulkan. After the simulation step another semaphore is signaled, or in the case of the timeline semaphore the same one is increased by CUDA, such that the Vulkan part that is waiting on this semaphore can continue rendering with the updated vertex buffers.
    
    
    #ifdef _VK_TIMELINE_SEMAPHORE
        static uint64_t waitValue = 1;
        static uint64_t signalValue = 2;
    
        cudaExternalSemaphoreWaitParams waitParams = {};
        waitParams.flags = 0;
        waitParams.params.fence.value = waitValue;
    
        cudaExternalSemaphoreSignalParams signalParams = {};
        signalParams.flags = 0;
        signalParams.params.fence.value = signalValue;
        // Wait for vulkan to complete it's work
        checkCudaErrors(cudaWaitExternalSemaphoresAsync(&m_cudaTimelineSemaphore,
                                                        &waitParams, 1, m_stream));
        // Now step the simulation, call CUDA kernel
        m_sim.stepSimulation(time, m_stream);
        // Signal vulkan to continue with the updated buffers
        checkCudaErrors(cudaSignalExternalSemaphoresAsync(
            &m_cudaTimelineSemaphore, &signalParams, 1, m_stream));
    
        waitValue += 2;
        signalValue += 2;
    #else
        cudaExternalSemaphoreWaitParams waitParams = {};
        waitParams.flags = 0;
        waitParams.params.fence.value = 0;
    
        cudaExternalSemaphoreSignalParams signalParams = {};
        signalParams.flags = 0;
        signalParams.params.fence.value = 0;
    
        // Wait for vulkan to complete it's work
        checkCudaErrors(cudaWaitExternalSemaphoresAsync(&m_cudaWaitSemaphore,
                                                        &waitParams, 1, m_stream));
        // Now step the simulation, call CUDA kernel
        m_sim.stepSimulation(time, m_stream);
        // Signal vulkan to continue with the updated buffers
        checkCudaErrors(cudaSignalExternalSemaphoresAsync(
            &m_cudaSignalSemaphore, &signalParams, 1, m_stream));
    #endif /* _VK_TIMELINE_SEMAPHORE */
    

#### 4.19.2.1.10. OpenGL Interoperability

Traditional OpenGL-CUDA interop as outlined in [OpenGL Interoperability](#opengl-interoperability) works by CUDA directly consuming handles created in OpenGL. However, since OpenGL can also consume memory and synchronization objects created in Vulkan, there exists an alternative approach to doing OpenGL-CUDA interop. Essentially, memory and synchronization objects exported by Vulkan could be imported into both, OpenGL and CUDA, and then used to coordinate memory accesses between OpenGL and CUDA. Please refer to the following OpenGL extensions for further details on how to import memory and synchronization objects exported by Vulkan:

  * `GL_EXT_memory_object`

  * `GL_EXT_memory_object_fd`

  * `GL_EXT_memory_object_win32`

  * `GL_EXT_semaphore`

  * `GL_EXT_semaphore_fd`

  * `GL_EXT_semaphore_win32`


### 4.19.2.2. Direct3D Interoperability

Importing Direct3D[11|12] resources to CUDA is supported for Direct3D11 and Direct3D12. We are only looking at the Direct3D12, for Direct3D11 please refer to the CUDA programming guide 12.9.

#### 4.19.2.2.1. Matching Device LUIDs

When importing memory and synchronization objects exported by Direct3D12, they must be imported and mapped on the same device as they were created on. The CUDA device that corresponds to the Direct3D12 device on which the objects were created can be determined by comparing the LUID of a CUDA device with that of the Direct3D12 device, as shown in the following code sample. Note that the Direct3D12 device must not be created on a linked node adapter, i.e. the node count as returned by `ID3D12Device::GetNodeCount` must be 1.
    
    
    int getCudaDeviceForD3D12Device(ID3D12Device *d3d12Device) {
        LUID d3d12Luid = d3d12Device->GetAdapterLuid();
    
        int cudaDeviceCount;
        cudaGetDeviceCount(&cudaDeviceCount);
    
        for (int cudaDevice = 0; cudaDevice < cudaDeviceCount; cudaDevice++) {
            cudaDeviceProp deviceProp;
            cudaGetDeviceProperties(&deviceProp, cudaDevice);
            char *cudaLuid = deviceProp.luid;
    
            if (!memcmp(&d3d12Luid.LowPart, cudaLuid, sizeof(d3d12Luid.LowPart)) &&
                !memcmp(&d3d12Luid.HighPart, cudaLuid + sizeof(d3d12Luid.LowPart), sizeof(d3d12Luid.HighPart))) {
                return cudaDevice;
            }
        }
        return cudaInvalidDeviceId;
    }
    

#### 4.19.2.2.2. Importing Memory Objects

There are several different ways how to import memory objects from NT handles. Note that it is the application’s responsibility to close the NT handle when it is not required anymore. The NT handle holds a reference to the resource, so it must be explicitly freed before the underlying memory can be freed. When importing a Direct3D resource, the flag `cudaExternalMemoryDedicated` must be set as in the snippets below.

A shareable Direct3D12 heap memory object, created by setting the flag `D3D12_HEAP_FLAG_SHARED` in the call to `ID3D12Device::CreateHeap`, can be imported into CUDA using the NT handle associated with that object as shown below.
    
    
    cudaExternalMemory_t importD3D12HeapFromNTHandle(HANDLE handle, unsigned long long size) {
        cudaExternalMemory_t extMem = NULL;
        cudaExternalMemoryHandleDesc desc = {};
    
        memset(&desc, 0, sizeof(desc));
    
        desc.type = cudaExternalMemoryHandleTypeD3D12Heap;
        desc.handle.win32.handle = (void *)handle;
        desc.size = size;
    
        cudaImportExternalMemory(&extMem, &desc);
    
        // Input parameter 'handle' should be closed if it's not needed anymore
        CloseHandle(handle);
    
        return extMem;
    }
    

A shareable Direct3D12 heap memory object can also be imported using a named handle if one exists:
    
    
    cudaExternalMemory_t importD3D12HeapFromNamedNTHandle(LPCWSTR name, unsigned long long size) {
        cudaExternalMemory_t extMem = NULL;
        cudaExternalMemoryHandleDesc desc = {};
    
        memset(&desc, 0, sizeof(desc));
    
        desc.type = cudaExternalMemoryHandleTypeD3D12Heap;
        desc.handle.win32.name = (void *)name;
        desc.size = size;
    
        cudaImportExternalMemory(&extMem, &desc);
    
        return extMem;
    }
    

A shareable Direct3D12 committed resource, created by setting the flag `D3D12_HEAP_FLAG_SHARED` in the call to `D3D12Device::CreateCommittedResource`, can be imported into CUDA using the NT handle associated with that object as shown below. When importing a Direct3D12 committed resource, the flag `cudaExternalMemoryDedicated` must be set.
    
    
    cudaExternalMemory_t importD3D12CommittedResourceFromNTHandle(HANDLE handle, unsigned long long size) {
        cudaExternalMemory_t extMem = NULL;
        cudaExternalMemoryHandleDesc desc = {};
    
        memset(&desc, 0, sizeof(desc));
    
        desc.type = cudaExternalMemoryHandleTypeD3D12Resource;
        desc.handle.win32.handle = (void *)handle;
        desc.size = size;
        desc.flags |= cudaExternalMemoryDedicated;
    
        cudaImportExternalMemory(&extMem, &desc);
    
        // Input parameter 'handle' should be closed if it's not needed anymore
        CloseHandle(handle);
    
        return extMem;
    }
    

A shareable Direct3D12 committed resource can also be imported using a named handle if one exists as shown below.
    
    
    cudaExternalMemory_t importD3D12CommittedResourceFromNamedNTHandle(LPCWSTR name, unsigned long long size) {
        cudaExternalMemory_t extMem = NULL;
        cudaExternalMemoryHandleDesc desc = {};
    
        memset(&desc, 0, sizeof(desc));
    
        desc.type = cudaExternalMemoryHandleTypeD3D12Resource;
        desc.handle.win32.name = (void *)name;
        desc.size = size;
        desc.flags |= cudaExternalMemoryDedicated;
    
        cudaImportExternalMemory(&extMem, &desc);
    
        return extMem;
    }
    

#### 4.19.2.2.3. Mapping Buffers onto Imported Memory Objects

A device pointer can be mapped onto an imported memory object as shown below. The offset and size of the mapping must match that specified when creating the mapping using the corresponding Direct3D12 API. All mapped device pointers must be freed using `cudaFree()`.
    
    
    void * mapBufferOntoExternalMemory(cudaExternalMemory_t extMem, unsigned long long offset, unsigned long long size) {
        void *ptr = NULL;
        cudaExternalMemoryBufferDesc desc = {};
    
        memset(&desc, 0, sizeof(desc));
    
        desc.offset = offset;
        desc.size = size;
    
        cudaExternalMemoryGetMappedBuffer(&ptr, extMem, &desc);
    
        // Note: 'ptr' must eventually be freed using cudaFree()
        return ptr;
    }
    

#### 4.19.2.2.4. Mapping Mipmapped Arrays onto Imported Memory Objects

A CUDA mipmapped array can be mapped onto an imported memory object as shown below. The offset, dimensions, format and number of mip levels must match that specified when creating the mapping using the corresponding Direct3D12 API. Additionally, if the mipmapped array can be bound as a render target in Direct3D12, the flag `cudaArrayColorAttachment` must be set. All mapped mipmapped arrays must be freed using `cudaFreeMipmappedArray()`. The following code sample shows how to convert parameters into the corresponding CUDA parameters when mapping mipmapped arrays onto imported memory objects.
    
    
    cudaMipmappedArray_t mapMipmappedArrayOntoExternalMemory(cudaExternalMemory_t extMem, unsigned long long offset, cudaChannelFormatDesc *formatDesc, cudaExtent *extent, unsigned int flags, unsigned int numLevels) {
        cudaMipmappedArray_t mipmap = NULL;
        cudaExternalMemoryMipmappedArrayDesc desc = {};
    
        memset(&desc, 0, sizeof(desc));
    
        desc.offset = offset;
        desc.formatDesc = *formatDesc;
        desc.extent = *extent;
        desc.flags = flags;
        desc.numLevels = numLevels;
    
        // Note: 'mipmap' must eventually be freed using cudaFreeMipmappedArray()
        cudaExternalMemoryGetMappedMipmappedArray(&mipmap, extMem, &desc);
    
        return mipmap;
    }
    
    cudaChannelFormatDesc getCudaChannelFormatDescForDxgiFormat(DXGI_FORMAT dxgiFormat)
    {
        cudaChannelFormatDesc d;
    
        memset(&d, 0, sizeof(d));
    
        switch (dxgiFormat) {
            case DXGI_FORMAT_R8_UINT:            d.x = 8;  d.y = 0;  d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindUnsigned; break;
            case DXGI_FORMAT_R8_SINT:            d.x = 8;  d.y = 0;  d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindSigned;   break;
            case DXGI_FORMAT_R8G8_UINT:          d.x = 8;  d.y = 8;  d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindUnsigned; break;
            case DXGI_FORMAT_R8G8_SINT:          d.x = 8;  d.y = 8;  d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindSigned;   break;
            case DXGI_FORMAT_R8G8B8A8_UINT:      d.x = 8;  d.y = 8;  d.z = 8;  d.w = 8;  d.f = cudaChannelFormatKindUnsigned; break;
            case DXGI_FORMAT_R8G8B8A8_SINT:      d.x = 8;  d.y = 8;  d.z = 8;  d.w = 8;  d.f = cudaChannelFormatKindSigned;   break;
            case DXGI_FORMAT_R16_UINT:           d.x = 16; d.y = 0;  d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindUnsigned; break;
            case DXGI_FORMAT_R16_SINT:           d.x = 16; d.y = 0;  d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindSigned;   break;
            case DXGI_FORMAT_R16G16_UINT:        d.x = 16; d.y = 16; d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindUnsigned; break;
            case DXGI_FORMAT_R16G16_SINT:        d.x = 16; d.y = 16; d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindSigned;   break;
            case DXGI_FORMAT_R16G16B16A16_UINT:  d.x = 16; d.y = 16; d.z = 16; d.w = 16; d.f = cudaChannelFormatKindUnsigned; break;
            case DXGI_FORMAT_R16G16B16A16_SINT:  d.x = 16; d.y = 16; d.z = 16; d.w = 16; d.f = cudaChannelFormatKindSigned;   break;
            case DXGI_FORMAT_R32_UINT:           d.x = 32; d.y = 0;  d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindUnsigned; break;
            case DXGI_FORMAT_R32_SINT:           d.x = 32; d.y = 0;  d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindSigned;   break;
            case DXGI_FORMAT_R32_FLOAT:          d.x = 32; d.y = 0;  d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindFloat;    break;
            case DXGI_FORMAT_R32G32_UINT:        d.x = 32; d.y = 32; d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindUnsigned; break;
            case DXGI_FORMAT_R32G32_SINT:        d.x = 32; d.y = 32; d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindSigned;   break;
            case DXGI_FORMAT_R32G32_FLOAT:       d.x = 32; d.y = 32; d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindFloat;    break;
            case DXGI_FORMAT_R32G32B32A32_UINT:  d.x = 32; d.y = 32; d.z = 32; d.w = 32; d.f = cudaChannelFormatKindUnsigned; break;
            case DXGI_FORMAT_R32G32B32A32_SINT:  d.x = 32; d.y = 32; d.z = 32; d.w = 32; d.f = cudaChannelFormatKindSigned;   break;
            case DXGI_FORMAT_R32G32B32A32_FLOAT: d.x = 32; d.y = 32; d.z = 32; d.w = 32; d.f = cudaChannelFormatKindFloat;    break;
            default: assert(0);
        }
        return d;
    }
    
    cudaExtent getCudaExtentForD3D12Extent(UINT64 width, UINT height, UINT16 depthOrArraySize, D3D12_SRV_DIMENSION d3d12SRVDimension) {
        cudaExtent e = { 0, 0, 0 };
    
        switch (d3d12SRVDimension) {
            case D3D12_SRV_DIMENSION_TEXTURE1D:        e.width = width; e.height = 0;      e.depth = 0;                break;
            case D3D12_SRV_DIMENSION_TEXTURE2D:        e.width = width; e.height = height; e.depth = 0;                break;
            case D3D12_SRV_DIMENSION_TEXTURE3D:        e.width = width; e.height = height; e.depth = depthOrArraySize; break;
            case D3D12_SRV_DIMENSION_TEXTURECUBE:      e.width = width; e.height = height; e.depth = depthOrArraySize; break;
            case D3D12_SRV_DIMENSION_TEXTURE1DARRAY:   e.width = width; e.height = 0;      e.depth = depthOrArraySize; break;
            case D3D12_SRV_DIMENSION_TEXTURE2DARRAY:   e.width = width; e.height = height; e.depth = depthOrArraySize; break;
            case D3D12_SRV_DIMENSION_TEXTURECUBEARRAY: e.width = width; e.height = height; e.depth = depthOrArraySize; break;
            default: assert(0);
        }
    
        return e;
    }
    
    unsigned int getCudaMipmappedArrayFlagsForD3D12Resource(D3D12_SRV_DIMENSION d3d12SRVDimension, D3D12_RESOURCE_FLAGS d3d12ResourceFlags, bool allowSurfaceLoadStore) {
        unsigned int flags = 0;
    
        switch (d3d12SRVDimension) {
            case D3D12_SRV_DIMENSION_TEXTURECUBE:      flags |= cudaArrayCubemap;                    break;
            case D3D12_SRV_DIMENSION_TEXTURECUBEARRAY: flags |= cudaArrayCubemap | cudaArrayLayered; break;
            case D3D12_SRV_DIMENSION_TEXTURE1DARRAY:   flags |= cudaArrayLayered;                    break;
            case D3D12_SRV_DIMENSION_TEXTURE2DARRAY:   flags |= cudaArrayLayered;                    break;
            default: break;
        }
    
        if (d3d12ResourceFlags & D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET) {
            flags |= cudaArrayColorAttachment;
        }
        if (allowSurfaceLoadStore) {
            flags |= cudaArraySurfaceLoadStore;
        }
    
        return flags;
    }
    

#### 4.19.2.2.5. Importing Synchronization Objects

A shareable Direct3D12 fence object, created by setting the flag `D3D12_FENCE_FLAG_SHARED` in the call to `ID3D12Device::CreateFence`, can be imported into CUDA using the NT handle associated with that object as shown below. Note that it is the application’s responsibility to close the handle when it is not required anymore. The NT handle holds a reference to the resource, so it must be explicitly freed before the underlying semaphore can be freed.
    
    
    cudaExternalSemaphore_t importD3D12FenceFromNTHandle(HANDLE handle) {
        cudaExternalSemaphore_t extSem = NULL;
        cudaExternalSemaphoreHandleDesc desc = {};
    
        memset(&desc, 0, sizeof(desc));
    
        desc.type = cudaExternalSemaphoreHandleTypeD3D12Fence;
        desc.handle.win32.handle = handle;
    
        cudaImportExternalSemaphore(&extSem, &desc);
    
        // Input parameter 'handle' should be closed if it's not needed anymore
        CloseHandle(handle);
    
        return extSem;
    }
    

A shareable Direct3D12 fence object can also be imported using a named handle if one exists as shown below.
    
    
    cudaExternalSemaphore_t importD3D12FenceFromNamedNTHandle(LPCWSTR name) {
        cudaExternalSemaphore_t extSem = NULL;
        cudaExternalSemaphoreHandleDesc desc = {};
     
        memset(&desc, 0, sizeof(desc));
    
        desc.type = cudaExternalSemaphoreHandleTypeD3D12Fence;
        desc.handle.win32.name = (void *)name;
    
        cudaImportExternalSemaphore(&extSem, &desc);
    
        return extSem;
    }
    

#### 4.19.2.2.6. Signaling/Waiting on Imported Synchronization Objects

Once the semaphores with fences have been imported from Direct3D12 they can be signaled and waited on.

Signaling a fence object sets its value. The corresponding wait that waits on this signal must be issued in Direct3D12. Note that the wait that waits on this signal must be issued after this signal has been issued.
    
    
    void signalExternalSemaphore(cudaExternalSemaphore_t extSem, unsigned long long value, cudaStream_t stream) {
        cudaExternalSemaphoreSignalParams params = {};
    
        memset(&params, 0, sizeof(params));
    
        params.params.fence.value = value;
    
        cudaSignalExternalSemaphoresAsync(&extSem, &params, 1, stream);
    }
    

A fence object waits until its value becomes equal or greater than to the specified value. The corresponding signal that it is waiting on must be issued in Direct3D12. Note that, the signal must be issued before this wait can be issued.
    
    
    void waitExternalSemaphore(cudaExternalSemaphore_t extSem, unsigned long long value, cudaStream_t stream) {
        cudaExternalSemaphoreWaitParams params = {};
    
        memset(&params, 0, sizeof(params));
    
        params.params.fence.value = value;
    
        cudaWaitExternalSemaphoresAsync(&extSem, &params, 1, stream);
    }
    

### 4.19.2.3. NVIDIA Software Communication Interface Interoperability (NVSCI)

NvSciBuf and NvSciSync are interfaces developed for serving the following purposes:

  * NvSciBuf: Allows applications to allocate and exchange buffers in memory

  * NvSciSync: Allows applications to manage synchronization objects at operation boundaries


More details on these interfaces are available at: <https://docs.nvidia.com/drive>.

#### 4.19.2.3.1. Importing Memory Objects

For allocating an NvSciBuf object compatible with a given CUDA device, the corresponding GPU id must be set with `NvSciBufGeneralAttrKey_GpuId` in the NvSciBuf attribute list as shown below. Optionally, applications can specify the following attributes -

  * `NvSciBufGeneralAttrKey_NeedCpuAccess`: Specifies if CPU access is required for the buffer

  * `NvSciBufRawBufferAttrKey_Align`: Specifies the alignment requirement of `NvSciBufType_RawBuffer`

  * `NvSciBufGeneralAttrKey_RequiredPerm`: Different access permissions can be configured for different UMDs per NvSciBuf memory object instance. For example, to provide the GPU with read-only access permissions to the buffer, create a duplicate NvSciBuf object using `NvSciBufObjDupWithReducePerm()` with `NvSciBufAccessPerm_Readonly` as the input parameter. Then import this newly created duplicate object with reduced permission into CUDA as shown

  * `NvSciBufGeneralAttrKey_EnableGpuCache`: To control GPU L2 cacheability

  * `NvSciBufGeneralAttrKey_EnableGpuCompression`: To specify GPU compression


Note

For more details on these attributes and their valid input options, refer to NvSciBuf Documentation.

The following code snippet illustrates their sample usage.
    
    
    NvSciBufObj createNvSciBufObject() {
       // Raw Buffer Attributes for CUDA
        NvSciBufType bufType = NvSciBufType_RawBuffer;
        uint64_t rawsize = SIZE;
        uint64_t align = 0;
        bool cpuaccess_flag = true;
        NvSciBufAttrValAccessPerm perm = NvSciBufAccessPerm_ReadWrite;
    
        NvSciRmGpuId gpuid[] ={};
        CUuuid uuid;
        cuDeviceGetUuid(&uuid, dev));
    
        memcpy(&gpuid[0].bytes, &uuid.bytes, sizeof(uuid.bytes));
        // Disable cache on dev
        NvSciBufAttrValGpuCache gpuCache[] = {{gpuid[0], false}};
        NvSciBufAttrValGpuCompression gpuCompression[] = {{gpuid[0], NvSciBufCompressionType_GenericCompressible}};
        // Fill in values
        NvSciBufAttrKeyValuePair rawbuffattrs[] = {
             { NvSciBufGeneralAttrKey_Types, &bufType, sizeof(bufType) },
             { NvSciBufRawBufferAttrKey_Size, &rawsize, sizeof(rawsize) },
             { NvSciBufRawBufferAttrKey_Align, &align, sizeof(align) },
             { NvSciBufGeneralAttrKey_NeedCpuAccess, &cpuaccess_flag, sizeof(cpuaccess_flag) },
             { NvSciBufGeneralAttrKey_RequiredPerm, &perm, sizeof(perm) },
             { NvSciBufGeneralAttrKey_GpuId, &gpuid, sizeof(gpuid) },
             { NvSciBufGeneralAttrKey_EnableGpuCache &gpuCache, sizeof(gpuCache) },
             { NvSciBufGeneralAttrKey_EnableGpuCompression &gpuCompression, sizeof(gpuCompression) }
        };
    
        // Create list by setting attributes
        err = NvSciBufAttrListSetAttrs(attrListBuffer, rawbuffattrs,
                sizeof(rawbuffattrs)/sizeof(NvSciBufAttrKeyValuePair));
    
        NvSciBufAttrListCreate(NvSciBufModule, &attrListBuffer);
    
        // Reconcile And Allocate
        NvSciBufAttrListReconcile(&attrListBuffer, 1, &attrListReconciledBuffer,
                           &attrListConflictBuffer)
        NvSciBufObjAlloc(attrListReconciledBuffer, &bufferObjRaw);
        return bufferObjRaw;
    }
    
    
    
    NvSciBufObj bufferObjRo; // Readonly NvSciBuf memory obj
    // Create a duplicate handle to the same memory buffer with reduced permissions
    NvSciBufObjDupWithReducePerm(bufferObjRaw, NvSciBufAccessPerm_Readonly, &bufferObjRo);
    return bufferObjRo;
    

The allocated NvSciBuf memory object can be imported in CUDA using the NvSciBufObj handle as shown below. Application should query the allocated NvSciBufObj for attributes required for filling CUDA External Memory Descriptor. Note that the attribute list and NvSciBuf objects should be maintained by the application. If the NvSciBuf object imported into CUDA is also mapped by other drivers, then based on `NvSciBufGeneralAttrKey_GpuSwNeedCacheCoherency` output attribute value the application must use NvSciSync objects (refer to [Importing Synchronization Objects](#importing-synchronization-objects-nvsci)) as appropriate barriers to maintain coherence between CUDA and the other drivers.

Note

For more details on how to allocate and maintain NvSciBuf objects refer to [NvSciBuf API Documentation.](https://developer.nvidia.com/docs/drive/drive-os/6.0.6/public/drive-os-linux-sdk/common/topics/nvsci/NvStreams1.html)
    
    
    cudaExternalMemory_t importNvSciBufObject (NvSciBufObj bufferObjRaw) {
    
        /*************** Query NvSciBuf Object **************/
        NvSciBufAttrKeyValuePair bufattrs[] = {
                    { NvSciBufRawBufferAttrKey_Size, NULL, 0 },
                    { NvSciBufGeneralAttrKey_GpuSwNeedCacheCoherency, NULL, 0 },
                    { NvSciBufGeneralAttrKey_EnableGpuCompression, NULL, 0 }
        };
        NvSciBufAttrListGetAttrs(retList, bufattrs,
            sizeof(bufattrs)/sizeof(NvSciBufAttrKeyValuePair)));
                    ret_size = *(static_cast<const uint64_t*>(bufattrs[0].value));
    
        // Note cache and compression are per GPU attributes, so read values for specific gpu by comparing UUID
        // Read cacheability granted by NvSciBuf
        int numGpus = bufattrs[1].len / sizeof(NvSciBufAttrValGpuCache);
        NvSciBufAttrValGpuCache[] cacheVal = (NvSciBufAttrValGpuCache *)bufattrs[1].value;
        bool ret_cacheVal;
        for (int i = 0; i < numGpus; i++) {
            if (memcmp(gpuid[0].bytes, cacheVal[i].gpuId.bytes, sizeof(CUuuid)) == 0) {
                ret_cacheVal = cacheVal[i].cacheability);
            }
        }
    
        // Read compression granted by NvSciBuf
        numGpus = bufattrs[2].len / sizeof(NvSciBufAttrValGpuCompression);
        NvSciBufAttrValGpuCompression[] compVal = (NvSciBufAttrValGpuCompression *)bufattrs[2].value;
        NvSciBufCompressionType ret_compVal;
        for (int i = 0; i < numGpus; i++) {
            if (memcmp(gpuid[0].bytes, compVal[i].gpuId.bytes, sizeof(CUuuid)) == 0) {
                ret_compVal = compVal[i].compressionType);
            }
        }
    
        /*************** NvSciBuf Registration With CUDA **************/
    
        // Fill up CUDA_EXTERNAL_MEMORY_HANDLE_DESC
        cudaExternalMemoryHandleDesc memHandleDesc;
        memset(&memHandleDesc, 0, sizeof(memHandleDesc));
        memHandleDesc.type = cudaExternalMemoryHandleTypeNvSciBuf;
        memHandleDesc.handle.nvSciBufObject = bufferObjRaw;
        // Set the NvSciBuf object with required access permissions in this step
        memHandleDesc.handle.nvSciBufObject = bufferObjRo;
        memHandleDesc.size = ret_size;
        cudaImportExternalMemory(&extMemBuffer, &memHandleDesc);
        return extMemBuffer;
     }
    

#### 4.19.2.3.2. Mapping Buffers onto Imported Memory Objects

A device pointer can be mapped onto an imported memory object as shown below. The offset and size of the mapping can be filled as per the attributes of the allocated `NvSciBufObj`. All mapped device pointers must be freed using `cudaFree()`.
    
    
    void * mapBufferOntoExternalMemory(cudaExternalMemory_t extMem, unsigned long long offset, unsigned long long size) {
        void *ptr = NULL;
        cudaExternalMemoryBufferDesc desc = {};
    
        memset(&desc, 0, sizeof(desc));
    
        desc.offset = offset;
        desc.size = size;
    
        cudaExternalMemoryGetMappedBuffer(&ptr, extMem, &desc);
    
        // Note: 'ptr' must eventually be freed using cudaFree()
        return ptr;
    }
    

#### 4.19.2.3.3. Mapping Mipmapped Arrays onto Imported Memory Objects

A CUDA mipmapped array can be mapped onto an imported memory object as shown below. The offset, dimensions and format can be filled as per the attributes of the allocated `NvSciBufObj`. All mapped mipmapped arrays must be freed using `cudaFreeMipmappedArray()`. The following code sample shows how to convert NvSciBuf attributes into the corresponding CUDA parameters when mapping mipmapped arrays onto imported memory objects.

Note

The number of mip levels must be 1.
    
    
    cudaMipmappedArray_t mapMipmappedArrayOntoExternalMemory(cudaExternalMemory_t extMem, unsigned long long offset, cudaChannelFormatDesc *formatDesc, cudaExtent *extent, unsigned int flags, unsigned int numLevels) {
        cudaMipmappedArray_t mipmap = NULL;
        cudaExternalMemoryMipmappedArrayDesc desc = {};
    
        memset(&desc, 0, sizeof(desc));
    
        desc.offset = offset;
        desc.formatDesc = *formatDesc;
        desc.extent = *extent;
        desc.flags = flags;
        desc.numLevels = numLevels;
    
        // Note: 'mipmap' must eventually be freed using cudaFreeMipmappedArray()
        cudaExternalMemoryGetMappedMipmappedArray(&mipmap, extMem, &desc);
    
        return mipmap;
    }
    

#### 4.19.2.3.4. Importing Synchronization Objects

NvSciSync attributes that are compatible with a given CUDA device can be generated using `cudaDeviceGetNvSciSyncAttributes()`. The returned attribute list can be used to create a `NvSciSyncObj` that is guaranteed compatibility with a given CUDA device.
    
    
    NvSciSyncObj createNvSciSyncObject() {
        NvSciSyncObj nvSciSyncObj
        int cudaDev0 = 0;
        int cudaDev1 = 1;
        NvSciSyncAttrList signalerAttrList = NULL;
        NvSciSyncAttrList waiterAttrList = NULL;
        NvSciSyncAttrList reconciledList = NULL;
        NvSciSyncAttrList newConflictList = NULL;
    
        NvSciSyncAttrListCreate(module, &signalerAttrList);
        NvSciSyncAttrListCreate(module, &waiterAttrList);
        NvSciSyncAttrList unreconciledList[2] = {NULL, NULL};
        unreconciledList[0] = signalerAttrList;
        unreconciledList[1] = waiterAttrList;
    
        cudaDeviceGetNvSciSyncAttributes(signalerAttrList, cudaDev0, CUDA_NVSCISYNC_ATTR_SIGNAL);
        cudaDeviceGetNvSciSyncAttributes(waiterAttrList, cudaDev1, CUDA_NVSCISYNC_ATTR_WAIT);
    
        NvSciSyncAttrListReconcile(unreconciledList, 2, &reconciledList, &newConflictList);
    
        NvSciSyncObjAlloc(reconciledList, &nvSciSyncObj);
    
        return nvSciSyncObj;
    }
    

An NvSciSync object (created as above) can be imported into CUDA using the NvSciSyncObj handle as shown below. Note that ownership of the NvSciSyncObj handle continues to lie with the application even after it is imported.
    
    
    cudaExternalSemaphore_t importNvSciSyncObject(void* nvSciSyncObj) {
        cudaExternalSemaphore_t extSem = NULL;
        cudaExternalSemaphoreHandleDesc desc = {};
    
        memset(&desc, 0, sizeof(desc));
    
        desc.type = cudaExternalSemaphoreHandleTypeNvSciSync;
        desc.handle.nvSciSyncObj = nvSciSyncObj;
    
        cudaImportExternalSemaphore(&extSem, &desc);
    
        // Deleting/Freeing the nvSciSyncObj beyond this point will lead to undefined behavior in CUDA
    
        return extSem;
    }
    

#### 4.19.2.3.5. Signaling/Waiting on Imported Synchronization Objects

An imported `NvSciSyncObj` object can be signaled as outlined below. Signaling NvSciSync backed semaphore object initializes the _fence_ parameter passed as input. This fence parameter is waited upon by a wait operation that corresponds to the aforementioned signal. Additionally, the wait that waits on this signal must be issued after this signal has been issued. If the flags are set to `cudaExternalSemaphoreSignalSkipNvSciBufMemSync` then memory synchronization operations (over all the imported NvSciBuf in this process) that are executed as a part of the signal operation by default are skipped. When `NvsciBufGeneralAttrKey_GpuSwNeedCacheCoherency` is FALSE, this flag should be set.
    
    
    void signalExternalSemaphore(cudaExternalSemaphore_t extSem, cudaStream_t stream, void *fence) {
        cudaExternalSemaphoreSignalParams signalParams = {};
    
        memset(&signalParams, 0, sizeof(signalParams));
    
        signalParams.params.nvSciSync.fence = (void*)fence;
        signalParams.flags = 0; //OR cudaExternalSemaphoreSignalSkipNvSciBufMemSync
    
        cudaSignalExternalSemaphoresAsync(&extSem, &signalParams, 1, stream);
    
    }
    

An imported `NvSciSyncObj` object can be waited upon as outlined below. Waiting on NvSciSync backed semaphore object waits until the input _fence_ parameter is signaled by the corresponding signaler. Additionally, the signal must be issued before the wait can be issued. If the flags are set to `cudaExternalSemaphoreWaitSkipNvSciBufMemSync` then memory synchronization operations (over all the imported NvSciBuf in this process) that are executed as a part of the signal operation by default are skipped. When `NvsciBufGeneralAttrKey_GpuSwNeedCacheCoherency` is FALSE, this flag should be set.
    
    
    void waitExternalSemaphore(cudaExternalSemaphore_t extSem, cudaStream_t stream, void *fence) {
         cudaExternalSemaphoreWaitParams waitParams = {};
    
        memset(&waitParams, 0, sizeof(waitParams));
    
        waitParams.params.nvSciSync.fence = (void*)fence;
        waitParams.flags = 0; //OR cudaExternalSemaphoreWaitSkipNvSciBufMemSync
    
        cudaWaitExternalSemaphoresAsync(&extSem, &waitParams, 1, stream);
    }
