
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, r2_score
trues = []
preds = []
trues_labels = []
preds_labels = []
preds_from_labels = []
for test_x in tqdm(test_dataset):
    trues.append(test_x[:,:input_dim,0].numpy())
    z,_ = model.encode(test_x[:,:input_dim,:])
    o = model.decode(z)
    preds.append(o[:,:, 0])
    trues_labels.append(test_x[:,input_dim:,0].numpy())
    preds_labels.append(z.numpy())
    o = model.decode(test_x[:,input_dim:,:])
    preds_from_labels.append(o[:,:,0])
    
trues = np.concatenate(trues)
preds = np.concatenate(preds)
trues_labels = np.concatenate(trues_labels)
preds_labels = np.concatenate(preds_labels)
preds_from_labels = np.concatenate(preds_from_labels)
    
print('R2 x vs decode(encode(x)): ', r2_score(trues, preds))
print('R2 x vs decode(y): ', r2_score(trues, preds_from_labels))
print('R2 y vs encode(x): ', r2_score(trues_labels, preds_labels))

print('RMSE x vs decode(encode(x)): ', mean_squared_error(trues, preds, squared=True))
print('RMSE x vs decode(y): ', mean_squared_error(trues, preds_from_labels, squared=True))
print('RMSE y vs encode(x):: ', mean_squared_error(trues_labels, preds_labels, squared=True))
