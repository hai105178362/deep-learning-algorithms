def run_write(arr):
    with open("hw4p2_submission.csv", 'w+') as f:
        f.write('Id,Predicted\n')
        for i, j in enumerate(arr):
            f.write(str(i) + ',' + str(j) + '\n')
    print("finished.")
