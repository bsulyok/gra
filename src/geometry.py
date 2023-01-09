from math import cos, cosh, sinh, acosh, pi


def native_disk_distance(r_1: float, theta_1: float, r_2: float, theta_2: float) -> float:
    if theta_1 == theta_2:
        return abs(r_1-r_2)
    return acosh(cosh(r_1)*cosh(r_2)-sinh(r_1)*sinh(r_2)*cos(pi-abs(pi-abs(theta_1-theta_2))))