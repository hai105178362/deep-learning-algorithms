import torch
from ctcdecode import CTCBeamDecoder
import helper.phoneme_list as PL
import net


def run_decoder(model, test_data, test_X, test_X_lens):
    phonemes = PL.PHONEME_LIST
    decoder = CTCBeamDecoder(['$'] * len(phonemes), beam_width=4, log_probs_input=True)
    with torch.no_grad():
        out, out_lens = model(test_X, test_X_lens)
    # print("decoding..")
    # print(out.shape, out_lens)
    # print(out.transpose(0, 1).shape)
    test_Y, _, _, test_Y_lens = decoder.decode(out.transpose(0, 1), out_lens)
    for i in range(len(test_data)):
        # visualize(test_data[i], out[:len(test_data[i]), i, :])
        # For the i-th sample in the batch, get the best output
        best_seq = test_Y[i, 0, :test_Y_lens[i, 0]]
        best_pron = ''.join(PL.PHONEME_MAP[i] for i in best_seq)
        # print(test_data[i], '->', best_pron)
        # print(best_pron)
        return best_pron




if __name__ == "__main__":
    testpath = "dataset.nosync/HW3P2_Data/wsj0_test.npy"
    testX = net.load_data(xpath=testpath, ypath=None)
    testX_lens = torch.Tensor([len(seq) for seq in testX]).to(net.DEVICE)
    testX = net.LinesDataset(testX)
    # test_loader = net.DataLoader(testX, shuffle=False, batch_size=net.BATCH_SIZE, collate_fn=net.collate_lines)
    test_loader = net.DataLoader(testX, shuffle=False, batch_size=1, collate_fn=net.collate_lines)
    M = net.Model(in_vocab=40, out_vocab=46, embed_size=40, hidden_size=64)
    M.load_state_dict(state_dict=torch.load('saved_models/5.pt', map_location=net.DEVICE))
    batch_id = 0
    ans = []
    print(len(testX))
    for inputs, targets in test_loader:
        batch_id += 1
        print(batch_id)
        # new_inputlen = testX_lens[(batch_id - 1) * net.BATCH_SIZE:batch_id * net.BATCH_SIZE]
        new_inputlen = testX_lens[(batch_id - 1) * 1:batch_id * 1]
        cur_result = run_decoder(model=M, test_data=inputs, test_X=inputs, test_X_lens=new_inputlen)
        ans.append(cur_result)

    # print(cur_result)
    # print(len(cur_result))
    print(ans)
    print(len(ans))
    with open("hw3p2_submission.csv", 'w+') as f:
        f.write('id,predicted\n')
        for i, j in enumerate(ans):
            f.write(str(i) + ',' + str(j) + '\n')
