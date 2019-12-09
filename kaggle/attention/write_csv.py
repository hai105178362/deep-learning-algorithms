import params as par
def run_write(arr):
    with open("raw_results/hw4p2_submission-ep{}.csv".format(str(par.config.model)), 'w+') as f:
        f.write('Id,Predicted\n')
        for i, j in enumerate(arr):
            f.write(str(i) + ',' + str(j) + '\n')
    print("finished.")
