#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <unistd.h>

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

struct TensorData {
  float *data;
  const std::string name;
  std::vector<int> shape;
};

@interface MultiArrayFeatureProvider : NSObject <MLFeatureProvider> {
  const std::vector<TensorData> *_inputs;
  NSSet *_featureNames;
}
@end

@implementation MultiArrayFeatureProvider

- (instancetype)initWithInputs:(const std::vector<TensorData> *)inputs {
  self = [super init];
  _inputs = inputs;
  for (auto &input : *_inputs) {
    if (input.name.empty()) {
      return nil;
    }
  }
  return self;
}

- (NSSet<NSString *> *)featureNames {
  if (_featureNames == nil) {
    NSMutableArray *names = [[NSMutableArray alloc] init];
    for (auto &input : *_inputs) {
      [names addObject:[NSString stringWithCString:input.name.c_str()
                                          encoding:[NSString defaultCStringEncoding]]];
    }
    _featureNames = [NSSet setWithArray:names];
  }
  return _featureNames;
}

- (MLFeatureValue *)featureValueForName:(NSString *)featureName {
  for (auto &input : *_inputs) {
    if ([featureName cStringUsingEncoding:NSUTF8StringEncoding] == input.name) {
      NSArray *shape = @[
        @(input.shape[0]),
        @(input.shape[1]),
        @(input.shape[2]),
        @(input.shape[3]),
      ];

      NSArray *strides = @[
        @(input.shape[1] * input.shape[2] * input.shape[3]),
        @(input.shape[2] * input.shape[3]),
        @(input.shape[3]),
        @1,
      ];

      NSError *error = nil;
      MLMultiArray *mlArray = [[MLMultiArray alloc] initWithDataPointer:input.data
                                                                  shape:shape
                                                               dataType:MLMultiArrayDataTypeFloat32
                                                                strides:strides
                                                            deallocator:(^(void *bytes){
                                                                        })error:&error];
      if (error != nil) {
        NSLog(@"Failed to create MLMultiArray for feature %@ error: %@", featureName,
              [error localizedDescription]);
        return nil;
      }
      auto *mlFeatureValue = [MLFeatureValue featureValueWithMultiArray:mlArray];
      return mlFeatureValue;
    }
  }

  NSLog(@"Feature %@ not found", featureName);
  return nil;
}
@end

int main(int argc, char *argv[]) {
  NSURL *modelURL = [NSURL URLWithString:[NSString stringWithCString:argv[1]
                                                            encoding:NSUTF8StringEncoding]];
  NSURL *compiledModelURL = [MLModel compileModelAtURL:modelURL error:nil];

  NSError *e;
  MLModel *mlmodel = [MLModel modelWithContentsOfURL:compiledModelURL error:&e];
  if (nil == mlmodel) {
    NSLog(@"hmmm, %@", e);
  }
  NSLog(@"modelDescription: %@", [mlmodel modelDescription]);

  NSArray<NSNumber *> *inputShape;
  MLMultiArrayDataType inputDataType;
  MLMultiArrayShapeConstraint *inputShapeConstraint;
  NSString *inputName;

  NSDictionary *inputDic = [[mlmodel modelDescription] inputDescriptionsByName];
  for (NSString *key in [inputDic allKeys]) {
    NSLog(@"key: %@", key);
    inputName = [inputDic[key] name];
    NSLog(@"value name: %@", [inputDic[key] name]);

    NSLog(@"value type: %ld", ((MLFeatureDescription *)inputDic[key]).type);
    if (((MLFeatureDescription *)inputDic[key]).type == MLFeatureTypeMultiArray) {
      NSLog(@"constraint: %@", [((MLFeatureDescription *)inputDic[key]) multiArrayConstraint]);
      inputShape = [[((MLFeatureDescription *)inputDic[key]) multiArrayConstraint] shape];
      inputDataType = [[((MLFeatureDescription *)inputDic[key]) multiArrayConstraint] dataType];
      inputShapeConstraint =
          [[((MLFeatureDescription *)inputDic[key]) multiArrayConstraint] shapeConstraint];
    }
  }
  std::vector<int> shape;
  for (int i = 0; i < [inputShape count]; i++) {
    shape.push_back([inputShape[i] intValue]);
  }

  // read raw input file into a char array;
  const char *INPUT_FILE = "grace_hopper.raw";
  int fd = open(INPUT_FILE, O_RDONLY);
  int input_size = 224 * 224 * 3;
  unsigned char *input_data_uint8 = (unsigned char *)malloc(input_size);
  ssize_t r_size = read(fd, input_data_uint8, input_size);
  NSLog(@"size read = %zd", r_size);

  // convert uint8 values to float32 values, because MLModel takes float32
  float *input_data_float = (float *)malloc(input_size * sizeof(float));
  for (int i = 0; i < input_size; i++) {
    // normoalize [0, 255] -> [-1.0, 1.0]
    input_data_float[i] = (float)((input_data_uint8[i] * 1.0 - 128) / 255.0);
  }
  NSLog(@"%f:%f:%f\n", input_data_float[64], input_data_float[65], input_data_float[66]);
  NSLog(@"input Name = %@", inputName);

  // convert float32 array to an input feature
  TensorData inputTensorData = {
      input_data_float, [inputName cStringUsingEncoding:[NSString defaultCStringEncoding]], shape};
  std::vector<TensorData> inputVector;
  inputVector.push_back(inputTensorData);

  MultiArrayFeatureProvider *f = [[MultiArrayFeatureProvider alloc]
      initWithInputs:(const std::vector<TensorData> *)&inputVector];

  // inference with the input feature
  MultiArrayFeatureProvider *o = (MultiArrayFeatureProvider *)[mlmodel predictionFromFeatures:f
                                                                                        error:&e];
  if (nil == e) {
    NSLog(@"Failed to predict with %@, error: %@", f, [e localizedDescription]);
  }
  NSLog(@"output names %@", [o featureNames]);
  NSLog(@"output Softmax: %@", [o featureValueForName:@"Softmax"]);
  MLMultiArray *outputArray = [[o featureValueForName:@"Softmax"] multiArrayValue];

  // get float array from output feature
  __block float *foo;
  [outputArray getBytesWithHandler:(^(const void *bytes, NSInteger size) {
                 NSLog(@"buffer size = %ld", size);
                 foo = (float *)bytes;
               })];

  // check if the output meet our expectation
  float max = 0;
  int index = 0;
  for (int i = 0; i < 1001; i++) {
    if (foo[i] > max) {
      max = foo[i];
      index = i;
    }
  }
  NSLog(@"%d\n", index);
}
