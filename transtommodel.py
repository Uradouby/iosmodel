from onnx_coreml import convert

def transtoMLModel():
    class_labels = ['air plane', 'automobile', 'bird', 'cat', 'deer', 'dog',
                    'frog', 'horse', 'ship', 'truck']
    model = convert(model='cifar10_net.onnx',
                    minimum_ios_deployment_target='13', image_input_names=['image'],
                    mode='classifier',
                    predicted_feature_name='classLabel',
                    class_labels=class_labels)
    model.save("cifar10_net.mlmodel")

if __name__ == "__main__":
    transtoMLModel()