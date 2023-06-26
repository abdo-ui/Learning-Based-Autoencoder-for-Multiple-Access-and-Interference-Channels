import json

import neptune
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm

from data import CustomDataLoader, generate_data
from models import RTNMAC, RTNMU
from test_utils import (
    calculate_BER_from_BLER,
    calculate_BLER,
    plot_BER,
    plot_constellation_hist,
)
from utils import SaveBestModel, generate_result_path, parse_args, setup_seed


def main(params):
    PATH = generate_result_path(params)

    with open(f"{PATH}/parameters.json", "w") as parameters_file:
        json.dump(params, parameters_file)

    # Setup Neptune run
    if params["log_run"]:
        import credentials

        run = neptune.init_run(
            project=credentials.neptune_project,
            tags="Script Run",
            api_token=credentials.NEPTUNE_API_TOKEN,
            source_files=["**/*.py"],
        )
        run["config/parameters"] = params

    setup_seed(params["seed"])

    # Generate datasets and create data loaders
    train_data1, train_labels1 = generate_data(params["train_num"], params["M"])
    train_data2, train_labels2 = generate_data(params["train_num"], params["M"])

    val_data1, val_labels1 = generate_data(params["val_num"], params["M"])
    val_data2, val_labels2 = generate_data(params["val_num"], params["M"])

    test_data1, test_labels1 = generate_data(params["test_num"], params["M"])
    test_data2, test_labels2 = generate_data(params["test_num"], params["M"])

    train_loader = CustomDataLoader(
        train_data1,
        train_labels1,
        train_data2,
        train_labels2,
        batch_size=params["batch_size"],
        shuffle=True,
    )

    test_loader = CustomDataLoader(
        test_data1, test_labels1, test_data2, test_labels2, batch_size=100_000
    )

    # Setup model, loss, and optimizer
    selected_model = RTNMAC if params["config"] == "mac" else RTNMU
    model = selected_model(
        params["M"],
        compressed_dim=params["L"],
        channel_mode=params["config"],
        fading=params["fading"],
        f_sigma=params["fading_sigma"],
    )
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=params["lr"])
    save_best = SaveBestModel(path=PATH, name=params["name"])
    total_loss = max if params["loss"] == "max" else lambda x1, x2: x1 + x2

    if params["log_run"]:
        run["config/model"] = type(model).__name__
        run["config/loss_fn"] = type(loss_fn).__name__
        run["config/optimizer"] = type(optimizer).__name__

    # Training Loop
    for epoch in range(1, params["epochs"] + 1):
        print(f"Epoch {epoch}/{params['epochs']}")
        model.train()
        for x1, y1, x2, y2 in tqdm(train_loader):
            x1_decoded, x2_decoded = model(x1, x2, params["A"])

            loss1 = loss_fn(x1_decoded, y1)
            loss2 = loss_fn(x2_decoded, y2)
            loss = total_loss(loss1, loss2)

            if params["log_run"]:
                run["training/batch/train_loss"].log(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_data1_decoded, val_data2_decoded = model(
                val_data1, val_data2, params["A"]
            )

            val_loss1 = loss_fn(val_data1_decoded, val_labels1)
            val_loss2 = loss_fn(val_data2_decoded, val_labels2)
            val_loss = total_loss(val_loss1, val_loss2)

            if params["log_run"]:
                run["training/batch/val_loss"].log(val_loss)

            save_best(val_loss, epoch, model)

    # Save best model weights in Neptune run:
    if params["log_run"]:
        run[f"io_files/artifacts/{params['name']}"].upload(save_best.file_path)

    # Testing
    load_best = torch.load(save_best.file_path)
    model.load_state_dict(load_best)

    power = torch.arange(1, 16, dtype=torch.float32)
    EpNo = 10 * torch.log10(power**2 / 4)
    BLER1 = torch.zeros_like(power)
    BLER2 = torch.zeros_like(power)
    BER1 = torch.zeros_like(power)
    BER2 = torch.zeros_like(power)
    Avg_BER = torch.zeros_like(power)
    Avg_BLER = torch.zeros_like(power)

    for i, A in enumerate(power):
        BLER1[i], BLER2[i] = calculate_BLER(model, test_loader, A)
        BER1[i] = calculate_BER_from_BLER(BLER1[i], params["k"])
        BER2[i] = calculate_BER_from_BLER(BLER2[i], params["k"])

        Avg_BLER[i] = (BLER1[i] + BLER2[i]) / 2
        Avg_BER[i] = (BER1[i] + BER2[i]) / 2

        print(
            f"peak intensity = {power[i]:5.2f} | BLER = {Avg_BLER[i]:.7f} | BER = {Avg_BER[i]:.7f}"
        )

        if params["log_run"]:
            run["testing/peak_intensity"].log(A)
            run["testing/EpNo(dB)"].log(EpNo[i])
            run["testing/BLER1"].log(BLER1[i])
            run["testing/BLER2"].log(BLER2[i])
            run["testing/Avg_BLER"].log(Avg_BLER[i])
            run["testing/BER1"].log(BER1[i])
            run["testing/BER2"].log(BER2[i])
            run["testing/Avg_BER"].log(Avg_BER[i])

    # BER Plot
    BER_fig = plot_BER(
        EpNo, Avg_BER, label=f'{params["name"]}({params["L"]},{params["k"]})'
    )
    BER_fig.savefig(f"{PATH}/BER_figure.jpg")

    if params["log_run"]:
        run["testing/BER_Curve"].upload(BER_fig)

    # Constellation Points
    const_pts = model.get_constellation_points(
        test_data1[:1000], test_data2[:1000], params["A"]
    )
    np.savetxt(
        f"{PATH}/constellation_points.csv", const_pts[0].detach().numpy(), delimiter=","
    )
    if params["log_run"]:
        run["testing/constellation_points"].upload(f"{PATH}/constellation_points.csv")

    const_pts_fig = plot_constellation_hist(const_pts[0].flatten().detach())
    const_pts_fig.savefig(f"{PATH}/Constellation_histogram.jpg")
    if params["log_run"]:
        run["testing/Constellation_histogram"].upload(const_pts_fig)

    # Create CSV file with all results
    with open(f"{PATH}/result.csv", "w") as result:
        result.write("A, EpNo, Avg_BER, Avg_BLER, BER_1, BER_2, BLER_1, BLER_2\n")
        for i, A in enumerate(power):
            result.write(
                f"{A}, {EpNo[i]}, {Avg_BER[i]}, {Avg_BLER[i]}, {BER1[i]}, {BER2[i]}, {BLER1[i]}, {BLER2[i]}\n"
            )

    if params["log_run"]:
        run["testing/results_table"].upload(f"{PATH}/result.csv")
        run.stop()


if __name__ == "__main__":
    args = parse_args()

    params = dict(
        seed=args.seed,
        log_run=args.logRun,
        epochs=args.epochs,
        batch_size=args.batchSize,
        lr=args.learningRate,
        name=args.model,
        config=args.config,
        description=f"{args.model}_{args.config}, R={args.k}/{args.L}, A={args.trainingA}, fading: {args.fading},  f_sgima={args.fadingSigma}",
        k=args.k,
        L=args.L,
        M=2**args.k,
        R=args.k / args.L,
        A=args.trainingA,
        train_num=args.trainNum,
        val_num=args.valNum,
        test_num=args.testNum,
        fading=args.fading,
        fading_sigma=args.fadingSigma,
        loss=args.lossType,
    )

    main(params)
