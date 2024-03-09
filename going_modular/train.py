def main(learning_rate: float, hidden_units: int, batch_size: int, num_epochs: int):
  """
  Trains a PyTorch image classification model using device-agnostic code.
  """

  import os
  import torch
  import data_setup, engine, model_builder, utils

  from torchvision import transforms

  # Setup directories
  train_dir = "data/pizza_steak_sushi/train"
  test_dir = "data/pizza_steak_sushi/test"

  # Setup target device
  device = "cuda" if torch.cuda.is_available() else "cpu"

  # Create transforms
  data_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
  ])

  # Create DataLoaders with help from data_setup.py
  train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
      train_dir=train_dir,
      test_dir=test_dir,
      transform=data_transform,
      batch_size=batch_size
  )

  # Create model with help from model_builder.py
  model = model_builder.TinyVGG(
      input_shape=3,
      hidden_units=hidden_units,
      output_shape=len(class_names)
  ).to(device)

  # Set loss and optimizer
  loss_fn = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(),
                              lr=learning_rate)

  # Start training with help from engine.py
  engine.train(model=model,
              train_dataloader=train_dataloader,
              test_dataloader=test_dataloader,
              loss_fn=loss_fn,
              optimizer=optimizer,
              epochs=num_epochs,
              device=device)

  # Save the model with help from utils.py
  utils.save_model(model=model,
                  target_dir="models",
                  model_name="05_going_modular_script_mode_tinyvgg_model_" + f"{learning_rate}_{batch_size}_{num_epochs}_hidden_units_" + ".pth")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Hyperparameters for the model')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='The learning rate for the model')
    parser.add_argument('--batch_size', type=int, default=32, help='The batch_size for training')
    parser.add_argument('--num_epochs', type=int, default=5, help='The number of epochs to run the model')
    parser.add_argument('--hidden_units', type=int, default=10, help='The number of hidden units')

    args = parser.parse_args()

    main(args.learning_rate, args.hidden_units, args.batch_size, args.num_epochs)