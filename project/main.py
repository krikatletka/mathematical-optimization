

from src.preparation import prepare_project
from src.lab1 import run_lab1
from src.lab2 import run_lab2
from src.lab3 import run_lab3
from src.lab4 import run_lab4
from src.lab5 import run_lab5
from src.metrics import psnr
from src.variant import LAMBDA0


def main():

    project = prepare_project()

    lab1_result = run_lab1(
        clean=project.clean,
        noisy=project.noisy,
    )

    lab2_result = run_lab2(
        project.clean,
        project.noisy,
        lab1_result.a,
        lab1_result.b,
    )


    lab3_result = run_lab3(
        clean=project.clean,
        noisy=project.noisy,
        gaussian_sigma=lab2_result.x,
    )

    lab4_result = run_lab4(
        clean=project.clean,
        noisy=project.noisy,
        lambda0=LAMBDA0,
    )

    print("\nPROJECT COMPLETE")

    print(
        "Lab 1 interval = "
        f"[{lab1_result.a:.6f}; "
        f"{lab1_result.b:.6f}]"
    )

    print(
        "Lab 2 optimal sigma = "
        f"{lab2_result.x:.9f}"
    )

    print(
        "Lab 3 optimal parameters = "
        f"({lab3_result['theta'][0]:.6f}; "
        f"{lab3_result['theta'][1]:.6f})"
    )

    print(
        "Lab 3 restored PSNR = "
        f"{lab3_result['psnr']:.6f} dB"
    )

    lab4_psnr = psnr(
        lab4_result["image"],
        project.clean,
    )

    print(
        "Lab 4 optimal lambda = "
        f"{lab4_result['lambda']:.9f}"
    )

    print(
        "Lab 4 restored PSNR = "
        f"{lab4_psnr:.6f} dB"
    )

    lab5_result = run_lab5(
        clean=project.clean,
        raw_noisy=project.raw_noisy,
        lambda_value=lab4_result["lambda"],
    )

    print(
        "Lab 5 restored PSNR = "
        f"{lab5_result['psnr']:.6f} dB"
    )

    


if __name__ == "__main__":
    main()