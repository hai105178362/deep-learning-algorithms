import torch
from ctcdecode import CTCBeamDecoder
from torch.utils.data import Dataset, DataLoader
import helper.phoneme_list as PL
import stringdist
import net


def run_decoder(model, test_data, test_X):
    inputlen = torch.IntTensor([len(seq) for seq in test_X]).to(net.DEVICE)
    phonemes = [' '] + PL.PHONEME_MAP
    decoder = CTCBeamDecoder(['$'] * (len(phonemes)), beam_width=100, log_probs_input=True)
    with torch.no_grad():
        out, out_lens = model(test_X, inputlen)
    test_Y, _, _, test_Y_lens = decoder.decode(out.transpose(0, 1), out_lens)
    for i in range(len(test_data)):
        # For the i-th sample in the batch, get the best output
        best_seq = test_Y[i, 0, :test_Y_lens[i, 0]]
        # print(best_seq)
        # exit()
        best_pron = ''.join(phonemes[i + 1] for i in best_seq)
        # print(test_data[i], '->', best_pron)
        # print(best_pron)
        return best_pron


if __name__ == "__main__":
    mode = "test"
    if mode == "test":
        testpath = "dataset.nosync/HW3P2_Data/wsj0_test.npy"
        testX = net.load_data(xpath=testpath, ypath=None)
        testX = net.LinesDataset(testX)
        inputs = list(testX)
        for i in range(len(inputs)):
            inputs[i] = torch.cat(inputs[i])
        inputlen = torch.IntTensor([len(seq) for seq in inputs]).to(net.DEVICE)
        print(len(inputs))
        print(inputlen)
        exit()
        test_loader = net.DataLoader(inputs, shuffle=False, batch_size=1)
        M = net.Model(in_vocab=40, out_vocab=47, hidden_size=net.HIDDEN_SIZE)
        M.load_state_dict(state_dict=torch.load('saved_models/6:21-4.pt', map_location=net.DEVICE))
        batch_id = 0
        ans = []
        print(len(testX))
        for inputs in test_loader:
            batch_id += 1
            # new_inputlen = testX_lens[(batch_id - 1) * net.BATCH_SIZE:batch_id * net.BATCH_SIZE]
            cur_result = run_decoder(model=M, test_data=inputs, test_X=inputs)
            print("{}:{}".format(batch_id, cur_result))
            ans.append(cur_result)
        print(ans)
        print(len(ans))
        with open("hw3p2_submission.csv", 'w+') as f:
            f.write('id,predicted\n')
            for i, j in enumerate(ans):
                f.write(str(i) + ',' + str(j) + '\n')
    else:
        valxpath = "dataset.nosync/HW3P2_Data/wsj0_dev.npy"
        valypath = "dataset.nosync/HW3P2_Data/wsj0_dev_merged_labels.npy"
        valX, valY = net.load_data(xpath=valxpath, ypath=valypath)
        for i in range(len(valY)):
            valY[i] = torch.IntTensor(valY[i]).to(net.DEVICE)
        # valX_lens = torch.Tensor([len(seq) for seq in valX]).to(net.DEVICE)
        # valY_lens = torch.IntTensor([len(seq) for seq in valY]).to(net.DEVICE)
        valX = net.LinesDataset(valX)
        inputs = list(valX)
        for i in range(len(inputs)):
            inputs[i] = torch.cat(inputs[i])
        val_loader = DataLoader(inputs, shuffle=False, batch_size=1)
        M = net.Model(in_vocab=40, out_vocab=47, hidden_size=net.HIDDEN_SIZE)
        M.load_state_dict(state_dict=torch.load('saved_models/6:21-4.pt', map_location=net.DEVICE))
        batch_id = 0
        ans = []
        n = 0
        tot_levd = 0
        for inputs in val_loader:
            batch_id += 1
            # new_inputlen = testX_lens[(batch_id - 1) * net.BATCH_SIZE:batch_id * net.BATCH_SIZE]
            new_inputlen = torch.IntTensor([len(seq) for seq in inputs]).to(net.DEVICE)
            ref_result = ''.join(PL.PHONEME_MAP[i] for i in valY[n])
            cur_result = run_decoder(model=M, test_data=inputs, test_X=inputs)
            n += 1
            print(cur_result, ref_result)
            lev_distance = stringdist.levenshtein(cur_result, ref_result)
            tot_levd += lev_distance
            print(lev_distance)
        print("Average Lev_distance:{}".format(tot_levd / len(valX)))
