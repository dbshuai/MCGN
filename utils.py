# encoding:utf-8
import numpy as np
import pickle
import random
import torch
import torch.utils.data as data

vocab_polarity_tags = {'O': 0, 'POSITIVE': 1, 'NEUTRAL': 2, 'NEGATIVE': 3, 'CONFLICT': 4}
vocab_aspect_tags = {'O': 0, 'B_AP': 1, 'I_AP': 2}
vocab_opinion_tags = {'O': 0, 'B_AP': 1, 'I_AP': 2}


def devide_ae(ae_seq):
    opinion_seq = []
    aspect_seq = []
    for seq in ae_seq:
        o_seq = []
        a_seq = []
        for s in seq:
            if s == 0:
                o_seq.append(0)
                a_seq.append(0)
            if s == 1:
                o_seq.append(0)
                a_seq.append(1)
            if s == 2:
                o_seq.append(0)
                a_seq.append(2)
            if s == 3:
                o_seq.append(1)
                a_seq.append(0)
            if s == 4:
                o_seq.append(2)
                a_seq.append(0)
        opinion_seq.append(o_seq)
        aspect_seq.append(a_seq)
    return aspect_seq, opinion_seq

#去掉conflict情感的aspect
def devide_ae_(ae_seq, polarity_seq):
    opinion_seq = []
    aspect_seq = []
    polarity_seq_new = []
    for ao_seq, pol_seq in zip(ae_seq, polarity_seq):
        o_seq = []
        a_seq = []
        p_seq = []
        for s, t in zip(ao_seq, pol_seq):
            if t != 4:
                if s == 0:
                    o_seq.append(0)
                    a_seq.append(0)
                if s == 1:
                    o_seq.append(0)
                    a_seq.append(1)
                if s == 2:
                    o_seq.append(0)
                    a_seq.append(2)
                if s == 3:
                    o_seq.append(1)
                    a_seq.append(0)
                if s == 4:
                    o_seq.append(2)
                    a_seq.append(0)
                p_seq.append(t)
            else:
                o_seq.append(0)
                a_seq.append(0)
                p_seq.append(0)
        opinion_seq.append(o_seq)
        aspect_seq.append(a_seq)
        polarity_seq_new.append(p_seq)
    return aspect_seq, opinion_seq, polarity_seq_new


def eval(gold_ae_seq, pred_ae_seq, gold_polarity_seq, pred_polarity_seq, seq_lens):
    #全体aspect
    aspect_lab_chunks = []
    aspect_lab_pred_chunks = []
    #不要conflict的aspect
    aspect_lab_chunks_ = []
    aspect_lab_pred_chunks_ = []

    opinion_lab_chunks = []
    opinion_lab_pred_chunks = []

    aspect_correct_preds, aspect_total_correct, aspect_total_preds = 0., 0., 0.
    opinion_correct_preds, opinion_total_correct, opinion_total_preds = 0., 0., 0.
    polarity_correct_preds, polarity_total_correct, polarity_total_preds = 0., 0., 0.

    gold_aspect_seq_, gold_opinion_seq_, gold_polarity_seq_ = devide_ae_(gold_ae_seq, gold_polarity_seq)
    pred_aspect_seq_, pred_opinion_seq_, pred_polarity_seq_ = devide_ae_(pred_ae_seq, pred_polarity_seq)
    gold_aspect_seq, gold_opinion_seq = devide_ae(gold_ae_seq)
    pred_aspect_seq, pred_opinion_seq = devide_ae(pred_ae_seq)

    for lab, lab_pred, length in zip(gold_aspect_seq, pred_aspect_seq, seq_lens):
        lab = lab[:length]
        lab_pred = lab_pred[:length]

        lab_chunks = get_chunks(lab, vocab_aspect_tags)
        aspect_lab_chunks.append(lab_chunks)
        lab_chunks = set(lab_chunks)

        lab_pred_chunks = get_chunks(lab_pred, vocab_aspect_tags)
        aspect_lab_pred_chunks.append(lab_pred_chunks)
        lab_pred_chunks = set(lab_pred_chunks)
        aspect_correct_preds += len(lab_chunks & lab_pred_chunks)
        aspect_total_preds += len(lab_pred_chunks)
        aspect_total_correct += len(lab_chunks)

    for lab, lab_pred, length in zip(gold_aspect_seq_, pred_aspect_seq_, seq_lens):
        lab = lab[:length]
        lab_pred = lab_pred[:length]

        lab_chunks = get_chunks(lab, vocab_aspect_tags)
        aspect_lab_chunks_.append(lab_chunks)

        lab_pred_chunks = get_chunks(lab_pred, vocab_aspect_tags)
        aspect_lab_pred_chunks_.append(lab_pred_chunks)


    for lab, lab_pred, length in zip(gold_opinion_seq, pred_opinion_seq, seq_lens):
        lab = lab[:length]
        lab_pred = lab_pred[:length]

        lab_chunks = get_chunks(lab, vocab_opinion_tags)
        opinion_lab_chunks.append(lab_chunks)
        lab_chunks = set(lab_chunks)

        lab_pred_chunks = get_chunks(lab_pred, vocab_opinion_tags)
        opinion_lab_pred_chunks.append(lab_pred_chunks)
        lab_pred_chunks = set(lab_pred_chunks)
        opinion_correct_preds += len(lab_chunks & lab_pred_chunks)
        opinion_total_preds += len(lab_pred_chunks)
        opinion_total_correct += len(lab_chunks)

    index = 0
    for lab, lab_pred, length in zip(gold_polarity_seq_, pred_polarity_seq_, seq_lens):
        lab = lab[:length]
        lab_pred = lab_pred[:length]
        lab_chunks = set(get_polaity_chunks(lab, vocab_polarity_tags, aspect_lab_chunks_[index]))
        lab_pred_chunks = set(get_polaity_chunks(lab_pred, vocab_polarity_tags, aspect_lab_pred_chunks_[index]))
        polarity_correct_preds += len(lab_chunks & lab_pred_chunks)
        polarity_total_preds += len(lab_pred_chunks)
        polarity_total_correct += len(lab_chunks)
        index += 1

    aspect_p, aspect_r, aspect_f1 = cacul_f1(aspect_correct_preds, aspect_total_preds, aspect_total_correct)
    opinion_p, opinion_r, opinion_f1 = cacul_f1(opinion_correct_preds, opinion_total_preds, opinion_total_correct)
    polarity_p, polarity_r, polarity_f1 = cacul_f1(polarity_correct_preds, polarity_total_preds,
                                                   polarity_total_correct)

    return aspect_f1, opinion_f1, polarity_f1


def get_chunk_type(tok, idx_to_tag):
    tag_name = idx_to_tag[tok]
    return tag_name.split('_')[-1]


def get_chunk_alpha(tok, idx_to_tag):
    tag_name = idx_to_tag[tok]
    return tag_name.split('_')[0]


def get_chunks(seq, vocab_tags):
    """
    Args:
        seq: [1, 0, 1, 1, 2, 0, 1, 2, 2, 0, 1] sequence of labels
        vocab_tags: {'O': 0, 'B_AP': 1, 'I_AP': 2}
    Returns:
        list of (chunk_type, chunk_start, chunk_end)
    Example:
        seq = [1, 0, 1, 1, 2, 0, 1, 2, 2, 0, 1]
        vocab_tags = {'O': 0, 'B_AP': 1, 'I_AP': 2}
        result = [('AP', 0, 1), ('AP', 2, 3), ('AP', 3, 5), ('AP', 6, 9), ('AP', 10, 11)]
    """
    default = vocab_tags["O"]
    idx_to_tag = {idx: tag for tag, idx in vocab_tags.items()}
    chunks = []
    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        # End of a chunk 1
        if tok == default and chunk_type is not None:
            # Add a chunk.
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None

        # End of a chunk + start of a chunk!
        elif tok != default:
            tok_chunk_type = get_chunk_type(tok, idx_to_tag)
            tok_chunk_alpha = get_chunk_alpha(tok, idx_to_tag)
            if chunk_type is None and tok_chunk_alpha == "B":
                chunk_type, chunk_start = tok_chunk_type, i
            elif chunk_type is not None and tok_chunk_type != chunk_type:
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = None, None
                if tok_chunk_alpha == "B":
                    chunk_type, chunk_start = tok_chunk_type, i
            elif chunk_type is not None and tok_chunk_type == chunk_type:
                if tok_chunk_alpha == "B":
                    chunk = (chunk_type, chunk_start, i)
                    chunks.append(chunk)
                    chunk_type, chunk_start = tok_chunk_type, i
        else:
            pass
    # end condition
    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq))
        chunks.append(chunk)
    return chunks


def get_polaity_chunks(seq, vocab_tags, aspect_lab_chunks):
    """
    Args:
        seq: [1, 0, 1, 1, 2, 0, 1, 2, 2, 0, 1] sequence of labels
        vocab_tags: {'O': 0, 'POSITIVE': 1, 'NEUTRAL': 2, 'NEGATIVE':3, 'CONFLICT':4}
        aspect_lab_chunks: [('AP', 0, 1), ('AP', 2, 3), ('AP', 3, 5), ('AP', 6, 9), ('AP', 10, 11)]
    Returns:
        list of (chunk_type, chunk_start, chunk_end)
    Example:
        seq = [1, 0, 1, 1, 2, 0, 1, 2, 2, 0, 1]
        vocab_tags = {'O': 0, 'POSITIVE': 1, 'NEUTRAL': 2, 'NEGATIVE':3, 'CONFLICT':4}
        aspect_lab_chunks = [('AP', 0, 1), ('AP', 2, 3), ('AP', 3, 5), ('AP', 6, 9), ('AP', 10, 11)]
        result = [('POSITIVE', 0, 1), ('POSITIVE', 2, 3), ('POSITIVE', 3, 5), ('NEUTRAL', 6, 9), ('POSITIVE', 10, 11)]
    """
    idx_to_tag = {idx: tag for tag, idx in vocab_tags.items()}
    default = vocab_tags["O"]
    chunks = []
    for i, chunk in enumerate(aspect_lab_chunks):
        segs = seq[chunk[1]:chunk[2]]
        counts = np.bincount(np.array(segs, dtype=int))
        counts = np.where(counts == max(counts))
        for indx in counts[0]:
            if default != indx:
                chunk_type = idx_to_tag[int(indx)]
                chunk = (chunk_type, chunk[1], chunk[2])
                chunks.append(chunk)
    return chunks


def cacul_f1(correct_preds, total_preds, total_correct):
    p = correct_preds / total_preds if total_preds > 0 else 0.
    r = correct_preds / total_correct if total_correct > 0 else 0.
    f1 = 2 * p * r / (p + r) if p > 0 and r > 0 else 0.
    return p, r, f1


def pkl_dump(your_object, des_file):
    with open(des_file, 'wb') as f:
        pickle.dump(your_object, f)


def pkl_load(pkl_file):
    with open(pkl_file, 'rb') as f:
        your_object = pickle.load(f)
    return your_object


def shuffle(list):
    for l in list:
        random.seed(12345)
        random.shuffle(l)


class DocDataset(data.Dataset):
    def __init__(self, doc_indexes, doc_labels, doc_domains, max_len):
        self.doc_indexes = doc_indexes
        self.doc_labels = doc_labels
        self.doc_domains = doc_domains
        self.max_len = max_len

    def get_padding_doc(self, doc_index, doc_label, doc_domain, max_len):
        num_words = len(doc_index)
        num_pad = max_len - num_words
        doc_index = np.asarray(doc_index[:max_len], dtype=np.int64)
        doc_label = np.asarray(doc_label, dtype=np.int64)
        doc_domain = np.asarray(doc_domain, dtype=np.int64)
        if num_pad > 0:
            doc_index = np.pad(doc_index, (0, num_pad), "constant", constant_values=(0, 0))
        return doc_index.astype(np.int64), doc_label.astype(np.int64), doc_domain.astype(np.int64), num_words

    def __getitem__(self, index):
        doc_index = self.doc_indexes[index]
        doc_label = self.doc_labels[index]
        doc_domain = self.doc_domains[index]
        doc_index, doc_label, doc_domain, length = self.get_padding_doc(doc_index, doc_label, doc_domain, self.max_len)
        return [doc_index, doc_label, doc_domain, length]

    def __len__(self):
        return len(self.doc_indexes)


class SentDataset(data.Dataset):
    # 为dataloader提供数据
    def __init__(self, sent_indexes,
                 ae_tags,
                 as_tags,
                 op_label_input,
                 ap_label_input,
                 max_len):
        self.sent_indexes = sent_indexes
        self.ae_tags = ae_tags
        self.as_tags = as_tags
        self.op_label_inputs = op_label_input
        self.ap_label_inputs = ap_label_input
        self.pad_max_len = max_len

    def get_padding_sent(self, sent_index, ae_tag, as_tag, op_label_input, ap_label_input, max_len):
        num_words = len(sent_index)
        num_pad = max_len - num_words
        sent_index = np.asarray(sent_index[:max_len], dtype=np.int64)
        as_tag = np.asarray(as_tag[:max_len], dtype=np.int64)
        ae_tag = np.asarray(ae_tag[:max_len], dtype=np.int64)
        op_label_input = np.asarray(op_label_input, dtype=np.float32)[..., :max_len]
        ap_label_input = np.asarray(ap_label_input, dtype=np.float32)[..., :max_len]
        mask = np.ones(num_words)[:max_len]
        if num_pad > 0:
            sent_index = np.pad(sent_index, (0, num_pad), "constant", constant_values=(0, 0))
            as_tag = np.pad(as_tag, (0, num_pad), "constant", constant_values=(0, 0))
            ae_tag = np.pad(ae_tag, (0, num_pad), "constant", constant_values=(0, 0))
            op_label_input = np.pad(op_label_input, ((0, num_pad), (0, 0)), "constant", constant_values=(0, 0))
            ap_label_input = np.pad(ap_label_input, ((0, num_pad), (0, 0)), "constant", constant_values=(0, 0))
            mask = np.pad(mask, (0, num_pad), "constant", constant_values=(0, 0))
        return sent_index.astype(np.int64), ae_tag.astype(np.int64), as_tag.astype(np.int64), op_label_input.astype(
            np.float32), ap_label_input.astype(np.float32), mask.astype(np.int64), num_words

    def __getitem__(self, index):
        sent_index = self.sent_indexes[index]
        ae_tag = self.ae_tags[index]
        as_tag = self.as_tags[index]
        op_label_input = self.op_label_inputs[index]
        ap_label_input = self.ap_label_inputs[index]
        sent_index, ae_tag, as_tag, op_label_input, ap_label_input, mask, length = self.get_padding_sent(
            sent_index, ae_tag, as_tag, op_label_input, ap_label_input, self.pad_max_len)
        return [sent_index, ae_tag, as_tag, op_label_input, ap_label_input, mask, length]

    def __len__(self):
        return len(self.sent_indexes)


if __name__ == '__main__':
    pred_aspect = torch.randn(3, 4)
    pred_aspect, i = torch.max(pred_aspect, dim=-1)
    print(pred_aspect, i)
    pred_sentiment = torch.Tensor([np.nan])
    print(torch.isnan(pred_sentiment))
