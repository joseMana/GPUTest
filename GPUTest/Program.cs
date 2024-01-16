using Cloo;
using System;
using System.Diagnostics;
using System.Linq;
using System.Management;
using System.Runtime.InteropServices;

namespace GPUTest
{
    class Program
    {
        static void Main()
        {
            //GetGPUInfo();
            AddUsingGPU(1000000);

            Console.ReadLine();
        }

        private static void AddUsingGPU(int vectorSize)
        {
            // Create input vectors
            float[] vectorA = Enumerable.Range(1, vectorSize).Select(i => (float)i).ToArray();
            float[] vectorB = Enumerable.Range(1, vectorSize).Select(i => (float)i * 2).ToArray();

            // Initialize OpenCL
            ComputeContext context = new ComputeContext(ComputeDeviceTypes.Gpu, new ComputeContextPropertyList(ComputePlatform.Platforms.First()), null, IntPtr.Zero);
            ComputeCommandQueue queue = new ComputeCommandQueue(context, context.Devices[0], ComputeCommandQueueFlags.None);

            // Create OpenCL buffers for input and output vectors
            ComputeBuffer<float> bufferA = new ComputeBuffer<float>(context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.UseHostPointer, vectorA);
            ComputeBuffer<float> bufferB = new ComputeBuffer<float>(context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.UseHostPointer, vectorB);
            ComputeBuffer<float> bufferResult = new ComputeBuffer<float>(context, ComputeMemoryFlags.WriteOnly, vectorSize);

            // Create and build the OpenCL program
            string kernelSource = @"
            __kernel void VectorAdd(__global const float* a, __global const float* b, __global float* result)
            {
                int i = get_global_id(0);
                result[i] = a[i] + b[i];
            }";

            ComputeProgram program = new ComputeProgram(context, kernelSource);
            program.Build(new[] { context.Devices[0] }, null, null, IntPtr.Zero);

            // Create the kernel and set its arguments
            ComputeKernel kernel = program.CreateKernel("VectorAdd");
            kernel.SetMemoryArgument(0, bufferA);
            kernel.SetMemoryArgument(1, bufferB);
            kernel.SetMemoryArgument(2, bufferResult);

            // Measure the time taken for the GPU vector addition
            Stopwatch stopwatch = new Stopwatch();
            stopwatch.Start();

            // Execute the kernel
            queue.ExecuteTask(kernel, null);
            queue.Finish();

            stopwatch.Stop();

            Console.WriteLine($"Time taken for GPU vector addition: {stopwatch.ElapsedMilliseconds} milliseconds");

            // Read the result back from the GPU
            float[] result = new float[vectorSize];
            GCHandle handle = GCHandle.Alloc(result, GCHandleType.Pinned);
            queue.Read(bufferResult, true, 0, vectorSize, handle.AddrOfPinnedObject(), null);
            handle.Free();

            // Display the result
            //Console.WriteLine("\nVector Addition Result:");
            //for (int i = 0; i < vectorSize; i++)
            //{
            //    Console.WriteLine($"Element {i}: {result[i]}");
            //}

            // Cleanup
            bufferA.Dispose();
            bufferB.Dispose();
            bufferResult.Dispose();
            kernel.Dispose();
            program.Dispose();
            queue.Dispose();
            context.Dispose();
        }

        private static void GetGPUInfo()
        {
            // Query for GPU information using WMI
            ManagementObjectSearcher searcher = new ManagementObjectSearcher("SELECT * FROM Win32_VideoController");

            Console.WriteLine("GPU Information:");

            foreach (ManagementObject obj in searcher.Get())
            {
                // Display GPU name and description
                Console.WriteLine($"Name: {obj["Name"]}");
                Console.WriteLine($"Description: {obj["Description"]}");

                // Check if Performance data is available
                if (obj["VideoProcessor"] != null)
                {
                    // Display Video Processor information
                    Console.WriteLine($"Video Processor: {obj["VideoProcessor"]}");
                }
                else
                {
                    Console.WriteLine("Performance information not available.");
                }

                Console.WriteLine();
            }

            // Check if any GPU was found
            if (searcher.Get().Count == 0)
            {
                Console.WriteLine("No GPU found on this system.");
            }
        }
    }

}
