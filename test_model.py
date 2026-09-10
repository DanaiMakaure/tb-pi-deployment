from tflite_runtime.interpreter import Interpreter

MODEL_PATH = "mobilenetv3_tb_pi.tflite"

print("Loading  model....")

interpreter=Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

print("Model loaded successfully!")

input_details=interpreter.get_input_details()
output_details=interpreter.get_output_details()

print("\nInput details:")
print(input_details)

print("\nOutput details:")
print(output_details)


