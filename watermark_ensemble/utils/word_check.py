from nltk.tokenize import NLTKWordTokenizer


word_tokenizer = NLTKWordTokenizer()


def keep_last_word(tokenizer, token_ids, stop_token_ids=None):
    """Extract the first complete word from a sequence of token IDs.

    Returns a dict with 'new_str' (the extracted text) and 'terminate' (bool).
    """
    def _find_valid_len(token_ids, stop_ids):
        list_ids = token_ids.tolist()
        for i, tid in enumerate(list_ids):
            if tid in stop_ids:
                return i
        return len(token_ids)

    if stop_token_ids is None:
        stop_token_ids = {tokenizer.eos_token_id}
    valid_len = _find_valid_len(token_ids, stop_token_ids)
    has_eos = valid_len < len(token_ids)
    current_str = tokenizer.decode(token_ids[:valid_len], skip_special_tokens=True)
    current_span_list = list(word_tokenizer.span_tokenize(current_str))

    if not current_span_list:
        return {"new_str": current_str if has_eos else "", "terminate": has_eos}
    elif current_span_list[0][1] == len(current_str):
        return {"new_str": current_str if has_eos else "", "terminate": has_eos}
    else:
        if has_eos and len(current_span_list) == 1:
            return {"new_str": current_str, "terminate": True}
        else:
            word_pos = current_span_list[0][1]
            result_str = current_str
            check_len = 1
            while check_len < valid_len:
                check_str = tokenizer.decode(token_ids[:check_len], skip_special_tokens=True)
                if len(check_str) >= word_pos:
                    result_str = check_str
                    break
                check_len += 1
            return {"new_str": result_str, "terminate": False}
