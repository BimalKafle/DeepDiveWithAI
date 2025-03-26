import os;
import pandas as pd
import torch
os.makedirs(os.path.join('data'),exist_ok=True)
data_file=os.path.join('data','user_contact.csv')

data=pd.read_csv(data_file)
print(data)
inputs = data.iloc[:, 0:3]
inputs=pd.get_dummies(inputs,dummy_na=True)
inputs=inputs.fillna(inputs.mean())
x=torch.tensor(inputs.to_numpy(dtype=float))


# Convert the encoded targets to a torch tensor
y = torch.tensor(x)

print(x)
print(y)
