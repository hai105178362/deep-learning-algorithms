import torch
from ctcdecode import CTCBeamDecoder
from torch.utils.data import Dataset, DataLoader
import helper.phoneme_list as PL
import stringdist
# import net
import netdev as net
# import net5layers as net


def run_decoder(model, inputs):
    inputlen = torch.IntTensor([len(seq) for seq in inputs]).to(net.DEVICE)
    phonemes = [' '] + PL.PHONEME_MAP
    decoder = CTCBeamDecoder(['$'] * (len(phonemes)), beam_width=200, log_probs_input=True)
    with torch.no_grad():
        out, out_lens = model(inputs, inputlen)
    test_Y, _, _, test_Y_lens = decoder.decode(out.transpose(0, 1), out_lens)
    for i in range(len(inputs)):
        # For the i-th sample in the batch, get the best output
        best_seq = test_Y[i, 0, :test_Y_lens[i, 0]]
        best_pron = ''.join(phonemes[i + 1] for i in best_seq)
        return best_pron

def collate_lines(seq_list):
    inputs = seq_list
    inputs = list(inputs)
    for i in range(len(inputs)):
        inputs[i] = torch.cat(inputs[i])
    return inputs


if __name__ == "__main__":
    # mode = "v"
    mode = "test"
    model_name = '2034-59'
    writename = model_name
    # model_name = 't4_models/196-24'
    # writename = model_name.split('/')[1]
    if mode == "test":
        testpath = "dataset.nosync/HW3P2_Data/wsj0_test.npy"
        testX = net.load_data(xpath=testpath, ypath=None)
        testX = net.LinesDataset(testX)
        inputs = list(testX)
        # inputlen = torch.IntTensor([len(seq) for seq in inputs]).to(net.DEVICE)
        test_loader = net.DataLoader(testX, shuffle=False, batch_size=1,collate_fn=collate_lines)
        M = net.Model(in_vocab=40, out_vocab=47, hidden_size=net.HIDDEN_SIZE)
        M.load_state_dict(state_dict=torch.load('saved_models/{}.pt'.format(model_name), map_location=net.DEVICE))
        batch_id = 0
        ans = []
        for inputs in test_loader:
            # new_inputlen = testX_lens[(batch_id - 1) * net.BATCH_SIZE:batch_id * net.BATCH_SIZE]
            cur_result = run_decoder(model=M,inputs=inputs)
            print("{},{}".format(batch_id, cur_result))
            batch_id += 1
            ans.append(cur_result)
        # print(ans)
        # print(len(ans))
        with open("hw3p2_submission-{}.csv".format(writename), 'w+') as f:
            f.write('id,predicted\n')
            for i, j in enumerate(ans):
                f.write(str(i) + ',' + str(j) + '\n')
        print("finished.")
    else:
        valxpath = "dataset.nosync/HW3P2_Data/wsj0_dev.npy"
        valypath = "dataset.nosync/HW3P2_Data/wsj0_dev_merged_labels.npy"
        valX, valY = net.load_data(xpath=valxpath, ypath=valypath)
        for i in range(len(valY)):
            valY[i] = torch.IntTensor(valY[i]).to(net.DEVICE)
        valX = net.LinesDataset(valX)
        val_loader = DataLoader(valX, shuffle=False, batch_size=1,collate_fn=collate_lines)
        M = net.Model(in_vocab=40, out_vocab=47, hidden_size=net.HIDDEN_SIZE)
        M.load_state_dict(state_dict=torch.load('saved_models/{}.pt'.format(model_name), map_location=net.DEVICE))
        # M.load_state_dict(state_dict=torch.load('saved_models/cnn_layers/1525_5.pt', map_location=net.DEVICE))
        batch_id = 0
        ans = []
        n = 0
        tot_levd = 0
        for inputs in val_loader:
            batch_id += 1
            # new_inputlen = testX_lens[(batch_id - 1) * net.BATCH_SIZE:batch_id * net.BATCH_SIZE]
            ref_result = ''.join(PL.PHONEME_MAP[i] for i in valY[n])
            # print(ref_result)
            cur_result = run_decoder(model=M, inputs=inputs)
            n += 1
            print(cur_result, ref_result)
            lev_distance = stringdist.levenshtein(cur_result, ref_result)
            tot_levd += lev_distance
            print(lev_distance)
        print("Average Lev_distance:{}".format(tot_levd / len(valX)))


