# ERAV2_Lit
Pytorch Lightning trial on Resnet

# CustomResnet18 trained on CIFAR10 using Pytorch Lightning

## Basic expectations
- Migrate Custom Resnet18 code from Pytorch to Pytorch Lightning first and then to Spaces such that:
  - Migrate model on Lightning from Pytorch
  - Use Gradio for deployment of Spaces app

- Spaces app has these features:
  - Ask the user whether he/she wants to see GradCAM images
    - How many
    - From which layer
    - Allow opacity change as well
  - Ask whether he/she wants to view misclassified images
    - How many
  - Allow users to upload new images
  - Provide 10 example images as well
  - Ask how many top classes are to be shown (make sure the user cannot enter more than 10)


## Reference to the Gradio spaces app for inference tests
- https://huggingface.co/spaces/Chintan-Shah/CustomResnet_Cifar10_App
- Follow this to test on huggingface spaces the outcome from the model
- 
## Results:
- Model: Resnet18
- Epochs: 20
- Optimizer: SGD
- Criteria: CrossEntropyLoss
- Scheduler: OneCycleLR
- Parameters: 11.2 M
- Training
  - Loss=0.1406
  - Accuracy=94.57%
- Validation
  - Average loss: 0.8546
  - Accuracy: 91.05%

## Some observations
  - TBD

## Logs
Epoch 19: 100%
 196/196 [00:51<00:00,  3.81it/s, v_num=0, train_loss=0.262, train_acc=0.938, val_loss=0.237, val_acc=0.909]
 
## OneCycleLR on SGD 
![image](https://github.com/ChintanShahDS/ERAV2_Lit/assets/94432132/092a7fff-b577-4c1b-9d21-e9d2e9b7c001)

## Train accuracy
![image](https://github.com/ChintanShahDS/ERAV2_Lit/assets/94432132/b600649e-a3a9-4885-8cf5-3b5ffee009dc)

## Train loss
![image](https://github.com/ChintanShahDS/ERAV2_Lit/assets/94432132/04b212d6-0d93-4e5b-826c-43dbc3241c15)

## Validation accuracy
![image](https://github.com/ChintanShahDS/ERAV2_Lit/assets/94432132/ed4a275c-c3e0-41d0-8a0f-8102e11f9602)

## Validation loss
![image](https://github.com/ChintanShahDS/ERAV2_Lit/assets/94432132/f008e9aa-e173-4d3c-9a6c-ffbf1aa48e3d)

## Misclassified images
![image](https://github.com/ChintanShahDS/ERAV2_Lit/assets/94432132/6779bb36-a7b4-403c-bfe3-bca64fdf0363)

## GradCAM Misclassified images
![image](https://github.com/ChintanShahDS/ERAV2_Lit/assets/94432132/dc7adef3-103b-4ebe-8ce3-980d11b6131c)





