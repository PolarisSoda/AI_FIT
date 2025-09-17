import torch

def pad_or_trim(x: torch.Tensor, target_len: int = 16):
    _,T,_,_ = x.shape

    if T == target_len:
        return x
    elif T < target_len:
        pad = x[:, -1:, :, :].repeat(1, target_len-T, 1, 1)
        return torch.cat([x, pad], dim=1)
    else:
        return x[:, :target_len, :, :]

def collate_fn(batch, num_states_per_ex, target_len=16):
    xs, ex_labels, conds = zip(*batch)
    xs = [pad_or_trim(x, target_len) for x in xs]
    xs = torch.stack(xs)  # [N,C,T,V,M]
    ex_labels = torch.stack(ex_labels)

    state_labels_per_ex = []
    for ex_idx, n_state in enumerate(num_states_per_ex):
        tmp = torch.zeros(len(batch), n_state, dtype=torch.float32)
        for i, (lbl, cond_vec) in enumerate(zip(ex_labels, conds)):
            if lbl.item() == ex_idx:
                tmp[i] = cond_vec
        state_labels_per_ex.append(tmp)
    return xs, ex_labels, state_labels_per_ex