#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

int main(int argc, char **argv) {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) {
            NSLog(@"No Metal device found.");
            return 2;
        }
        NSLog(@"Metal device: %@", dev.name);

        NSString *src =
            @"#include <metal_stdlib>\n"
             "using namespace metal;\n"
             "kernel void k(device double* out [[buffer(0)]],\n"
             "              uint gid [[thread_position_in_grid]]) {\n"
             "  out[gid] = 1.0;\n"
             "}\n";

        NSError *err = nil;
        MTLCompileOptions *opts = [MTLCompileOptions new];
        id<MTLLibrary> lib = [dev newLibraryWithSource:src options:opts error:&err];
        if (!lib) {
            NSLog(@"FP64 NOT supported: %@", err.localizedDescription);
            return 1;
        }

        id<MTLFunction> fn = [lib newFunctionWithName:@"k"];
        if (!fn) {
            NSLog(@"Library built but function not found; FP64 ambiguous.");
            return 3;
        }

        id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
        if (!pso) {
            NSLog(@"FP64 likely not supported at pipeline creation: %@", err.localizedDescription);
            return 1;
        }

        NSLog(@"FP64 appears supported on this GPU.");
        return 0;
    }
}
