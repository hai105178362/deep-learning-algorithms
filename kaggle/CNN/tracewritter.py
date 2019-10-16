import datetime

def log_title(allspec):
    d = datetime.datetime.today()
    record = "{}-{}-{}\n".format(d.day, d.hour, d.minute)
    cnn_logger = open("tracelog.txt", "a")
    cnn_logger.write("==========================================================================\n")
    cnn_logger.write(record)
    cnn_logger.write(allspec + "\n\n")
    cnn_logger.close()


def recordtrace(train_acc, train_loss, val_acc, val_loss, epoch):
        cnn_logger = open("tracelog.txt", "a")
        cnn_logger.write('Train Loss: {:.4f}\tTrain Accuracy: {:.4f}\tVal Loss: {:.4f}\tVal Accuracy: {:.4f} \t epoch: {}\n'.format(train_loss, train_acc, val_loss, val_acc, epoch))
        cnn_logger.close()
