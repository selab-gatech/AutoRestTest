def reason_type(first_param_body, second_param_body):
    is_same = True
    attributes = ["type",  "enum", "format"]

    # attributes = ["type", "minLength", "maxLength", "maximum", "minimum", "enum", "pattern", "format"]
    # check for type
    for attri in attributes:
        # pass for not having
        if not (attri in first_param_body) and not (attri in second_param_body):
            continue
        # check for consistent property
        if (attri in first_param_body and not (attri in second_param_body)) or (
                not (attri in first_param_body) and attri in second_param_body):
            is_same = False
            break
        # check for =
        if first_param_body[attri] != second_param_body[attri]:
            is_same = False
            break
    return is_same
