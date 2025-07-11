import os

# Get the absolute path of the current working directory
base_path = os.path.dirname(os.path.abspath(__file__))
print("Current script directory:", base_path)

# Show all items in that folder
print("Items in the directory:")
print(os.listdir(base_path))

# Now build and test dataset paths
real_path = os.path.join(base_path, "dataset", "dataset_real")
fake_path = os.path.join(base_path, "dataset", "dataset_fake")

print("Looking for real_path:", real_path)
print("Looking for fake_path:", fake_path)

print("Real path exists:", os.path.exists(real_path))
print("Fake path exists:", os.path.exists(fake_path))



