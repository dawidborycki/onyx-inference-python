import onnxruntime as ort
import numpy as np
import matplotlib.pyplot as plt
import wget, time, os, urllib
from tensorflow import keras

def download_model(model_name):
    if(not os.path.exists(model_name)):    
        base_url = 'https://github.com/onnx/models/raw/main/vision/classification/mnist/model/'
    
        url = urllib.parse.urljoin(base_url, model_name)
        
        wget.download(url)

def onnx_predict(onnx_session, input_name, output_name, test_images, test_labels, image_index, show_results):
    test_image = np.expand_dims(test_images[image_index], [0,1])

    onnx_pred = onnx_session.run([output_name], {input_name: test_image.astype('float32')})

    predicted_label = np.argmax(np.array(onnx_pred))
    actual_label = test_labels[image_index]

    if show_results:
        plt.figure()
        plt.xticks([])
        plt.yticks([])       
        plt.imshow(test_images[image_index], cmap=plt.cm.binary)
        
        plt.title('Actual: %s, predicted: %s' % (actual_label, predicted_label), fontsize=22)                
        plt.show()
    
    return predicted_label, actual_label

def measure_performance(onnx_session, input_name, output_name, test_images, test_labels, execution_count):
    start = time.time()    

    image_indices = np.random.randint(0, test_images.shape[0] - 1, execution_count)
    
    for i in range(1, execution_count):
        onnx_predict(onnx_session, input_name, output_name, test_images, test_labels, image_indices[i], False)
    
    computation_time = time.time() - start
    
    print('Computation time: %.3f ms' % (computation_time*1000))

if __name__ == "__main__":
    # Get model
    model_name = 'mnist-12.onnx'
    download_model(model_name)

    # Prepare inference session
    onnx_session = ort.InferenceSession(model_name)

    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name

    # Download and load MNIST dataset
    mnist_digits = keras.datasets.mnist
    _, (test_images, test_labels) = mnist_digits.load_data()

    # Run prediction
    image_index = np.random.randint(0, test_images.shape[0] - 1)
    onnx_predict(onnx_session, input_name, output_name, test_images, test_labels, image_index, True)

    # Evaluate execution time
    measure_performance(onnx_session, input_name, output_name, test_images, test_labels, execution_count=1000)
