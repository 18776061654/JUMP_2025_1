def calculate_score(d,posture_type):
    # 计算得分
    k,c = get_k_c(posture_type)
    score = k * d + c
    return score


def get_k_c(posture_type):
    if posture_type == 1:
        k = -0.36
        c = 83.68
        return k,c
    if posture_type == 2:
        k = 7.39
        c = 43.98
        return k,c
    if posture_type == 3:
        k = -0.23
        c = 77.39
        return k,c