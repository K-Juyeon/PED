import time
import datetime

from sklearn.metrics import f1_score, precision_score, recall_score

import train
import predict

if __name__ == '__main__':
    train_path = r'F:\PED\splitdata\train'
    val_path = r'F:\PED\splitdata\val'
    test_path = r'F:\PED\splitdata\test'
    modelss = 'cnn'
    optimizer = "sgd"
    epochs = 100
    early_stop = 5
    best_acc = 0
    batch_size = 128

    pt_path = f'.\\checkpoint\\{modelss}_{optimizer}.pt'

    start = time.time()

    for epoch in range(1, epochs + 1) :
        print(f"learning time Start : ", datetime.datetime.now())

        fine_tuning = train.FineTuning(train_path=train_path, val_path=val_path, modelss=modelss, epoch=epoch, optimizer=optimizer, batch_size=batch_size)
        trained_model = fine_tuning.training()
        best_acc, early_stop, all_targets, all_predictions = fine_tuning.validation(trained_model, early_stop=early_stop, epoch=epoch, best_acc=best_acc)

        print(f"learning time End : ", datetime.datetime.now())

        if early_stop > 5 :
            print("Early Stop")
            break

    print(f"learning time (100 epoch train+validate) : {time.time() - start}\n")

    f1 = f1_score(all_targets, all_predictions, average='macro')
    precision = precision_score(all_targets, all_predictions, average='macro')
    recall = recall_score(all_targets, all_predictions, average='macro')
    print(f"Validation - F1-Score: {f1: .4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

    predict_test = predict.Predict(test_path, modelss, pt_path, batch_size)
    predict_test.test()