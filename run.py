import os
import random
import argparse
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import torch

from models import DNN, RNN, LSTM, GRU, AttentionalLSTM, CNN
from utils import make_dirs, load_data, plot_full, data_loader, split_sequence_uni_step, split_sequence_multi_step
from utils import get_lr_scheduler, mean_percentage_error, mean_absolute_percentage_error, plot_pred_test

# Reproducibility #
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Device Configuration #
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def main(args):

    # Fix Seed #
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Weights and Plots Path #
    paths = [args.weights_path, args.plots_path, args.numpy_path]
    for path in paths:
        make_dirs(path)

    # Prepare Data #
    data = load_data(args.which_data)[[args.feature]]
    data = data.copy()

    # Plot Time-Series Data #
    if args.plot_full:
        plot_full(args.plots_path, data, args.feature)

    scaler = MinMaxScaler()
    data[args.feature] = scaler.fit_transform(data)

    # Split the Dataset #
    copied_data = data.copy().values

    if args.multi_step:
        X, y = split_sequence_multi_step(copied_data, args.seq_length, args.output_size)
        step = 'MultiStep'
    else:
        X, y = split_sequence_uni_step(copied_data, args.seq_length)
        step = 'SingleStep'

    train_loader, val_loader, test_loader = data_loader(X, y, args.train_split, args.test_split, args.batch_size)

    # Lists #
    train_losses, val_losses = list(), list()
    val_maes, val_mses, val_rmses, val_mapes, val_mpes, val_r2s = list(), list(), list(), list(), list(), list()
    test_maes, test_mses, test_rmses, test_mapes, test_mpes, test_r2s = list(), list(), list(), list(), list(), list()
    pred_tests, labels = list(), list()

    # Constants #
    best_val_loss = 100
    best_val_improv = 0

    # Prepare Network #
    if args.model == 'dnn':
        model = DNN(args.seq_length, args.hidden_size, args.output_size).to(device)
    elif args.model == 'cnn':
        model = CNN(args.seq_length, args.batch_size, args.output_size).to(device)
    elif args.model == 'rnn':
        model = RNN(args.input_size, args.hidden_size, args.num_layers, args.output_size).to(device)
    elif args.model == 'lstm':
        model = LSTM(args.input_size, args.hidden_size, args.num_layers, args.output_size, args.bidirectional).to(device)
    elif args.model == 'gru':
        model = GRU(args.input_size, args.hidden_size, args.num_layers, args.output_size).to(device)
    elif args.model == 'attentional':
        model = AttentionalLSTM(args.input_size, args.qkv, args.hidden_size, args.num_layers, args.output_size, args.bidirectional).to(device)
    else:
        raise NotImplementedError

    # Loss Function #
    criterion = torch.nn.MSELoss()

    # Optimizer #
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optim_scheduler = get_lr_scheduler(args.lr_scheduler, optim)

    # Train and Validation #
    if args.mode == 'train':

        # Train #
        print("Training {} using {} started with total epoch of {}.".format(model.__class__.__name__, step, args.num_epochs))

        for epoch in range(args.num_epochs):
            for i, (data, label) in enumerate(train_loader):

                # Prepare Data #
                data = data.to(device, dtype=torch.float32)
                label = label.to(device, dtype=torch.float32)

                # Forward Data #
                pred = model(data)

                # Calculate Loss #
                train_loss = criterion(pred, label)

                # Initialize Optimizer, Back Propagation and Update #
                optim.zero_grad()
                train_loss.backward()
                optim.step()

                # Add item to Lists #
                train_losses.append(train_loss.item())

            # Print Statistics #
            if (epoch+1) % args.print_every == 0:
                print("Epoch [{}/{}]".format(epoch+1, args.num_epochs))
                print("Train Loss {:.4f}".format(np.average(train_losses)))

            # Learning Rate Scheduler #
            optim_scheduler.step()

            # Validation #
            with torch.no_grad():
                for i, (data, label) in enumerate(val_loader):

                    # Prepare Data #
                    data = data.to(device, dtype=torch.float32)
                    label = label.to(device, dtype=torch.float32)

                    # Forward Data #
                    pred_val = model(data)

                    # Calculate Loss #
                    val_loss = criterion(pred_val, label)

                    if args.multi_step:
                        pred_val = np.mean(pred_val.detach().cpu().numpy(), axis=1)
                        label = np.mean(label.detach().cpu().numpy(), axis=1)
                    else:
                        pred_val, label = pred_val.cpu(), label.cpu()

                    # Calculate Metrics #
                    val_mae = mean_absolute_error(label, pred_val)
                    val_mse = mean_squared_error(label, pred_val, squared=True)
                    val_rmse = mean_squared_error(label, pred_val, squared=False)
                    val_mpe = mean_percentage_error(label, pred_val)
                    val_mape = mean_absolute_percentage_error(label, pred_val)
                    val_r2 = r2_score(label, pred_val)

                    # Add item to Lists #
                    val_losses.append(val_loss.item())
                    val_maes.append(val_mae.item())
                    val_mses.append(val_mse.item())
                    val_rmses.append(val_rmse.item())
                    val_mpes.append(val_mpe.item())
                    val_mapes.append(val_mape.item())
                    val_r2s.append(val_r2.item())

            if (epoch+1) % args.print_every == 0:

                # Print Statistics #
                print("Val Loss {:.4f}".format(np.average(val_losses)))
                print(" MAE : {:.4f}".format(np.average(val_maes)))
                print(" MSE : {:.4f}".format(np.average(val_mses)))
                print("RMSE : {:.4f}".format(np.average(val_rmses)))
                print(" MPE : {:.4f}".format(np.average(val_mpes)))
                print("MAPE : {:.4f}".format(np.average(val_mapes)))
                print(" R^2 : {:.4f}".format(np.average(val_r2s)))

                # Save the model only if validation loss decreased #
                curr_val_loss = np.average(val_losses)

                if curr_val_loss < best_val_loss:
                    best_val_loss = min(curr_val_loss, best_val_loss)
                    torch.save(model.state_dict(), os.path.join(args.weights_path, 'BEST_{}_using_{}.pkl'.format(model.__class__.__name__, step)))

                    print("Best model is saved!\n")
                    best_val_improv = 0

                elif curr_val_loss >= best_val_loss:
                    best_val_improv += 1
                    print("Best Validation has not improved for {} epochs.\n".format(best_val_improv))

    elif args.mode == 'test':

        # Load the Model Weight #
        model.load_state_dict(torch.load(os.path.join(args.weights_path, 'BEST_{}_using_{}.pkl'.format(model.__class__.__name__, step))))

        # Test #
        with torch.no_grad():
            for i, (data, label) in enumerate(test_loader):

                # Prepare Data #
                data = data.to(device, dtype=torch.float32)
                label = label.to(device, dtype=torch.float32)

                # Forward Data #
                pred_test = model(data)

                # Convert to Original Value Range #
                pred_test, label = pred_test.detach().cpu().numpy(), label.detach().cpu().numpy()

                pred_test = scaler.inverse_transform(pred_test)
                label = scaler.inverse_transform(label)

                if args.multi_step:
                    pred_test = np.mean(pred_test, axis=1)
                    label = np.mean(label, axis=1)

                pred_tests += pred_test.tolist()
                labels += label.tolist()

                # Calculate Loss #
                test_mae = mean_absolute_error(label, pred_test)
                test_mse = mean_squared_error(label, pred_test, squared=True)
                test_rmse = mean_squared_error(label, pred_test, squared=False)
                test_mpe = mean_percentage_error(label, pred_test)
                test_mape = mean_absolute_percentage_error(label, pred_test)
                test_r2 = r2_score(label, pred_test)

                # Add item to Lists #
                test_maes.append(test_mae.item())
                test_mses.append(test_mse.item())
                test_rmses.append(test_rmse.item())
                test_mpes.append(test_mpe.item())
                test_mapes.append(test_mape.item())
                test_r2s.append(test_r2.item())

            # Print Statistics #
            print("Test {} using {}".format(model.__class__.__name__, step))
            print(" MAE : {:.4f}".format(np.average(test_maes)))
            print(" MSE : {:.4f}".format(np.average(test_mses)))
            print("RMSE : {:.4f}".format(np.average(test_rmses)))
            print(" MPE : {:.4f}".format(np.average(test_mpes)))
            print("MAPE : {:.4f}".format(np.average(test_mapes)))
            print(" R^2 : {:.4f}".format(np.average(test_r2s)))

            # Plot Figure #
            plot_pred_test(pred_tests[:args.time_plot], labels[:args.time_plot], args.plots_path, args.feature, model, step)

            # Save Numpy files #
            np.save(os.path.join(args.numpy_path, '{}_using_{}_TestSet.npy'.format(model.__class__.__name__, step)), np.asarray(pred_tests))
            np.save(os.path.join(args.numpy_path, 'TestSet_using_{}.npy'.format(step)), np.asarray(labels))

    else:
        raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=7777, help='seed for reproducibility')
    parser.add_argument('--feature', type=str, default='Appliances', help='extract which feature for prediction')
    parser.add_argument('--multi_step', type=bool, default=False, help='multi-step or not')
    parser.add_argument('--seq_length', type=int, default=5, help='window size')
    parser.add_argument('--batch_size', type=int, default=128, help='mini-batch size')

    parser.add_argument('--plot_full', type=bool, default=False, help='plot full graph or not')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'inference'])

    parser.add_argument('--model', type=str, default='lstm', choices=['dnn', 'cnn', 'rnn', 'lstm', 'gru', 'attentional'])
    parser.add_argument('--input_size', type=int, default=1, help='input_size')
    parser.add_argument('--hidden_size', type=int, default=10, help='hidden_size')
    parser.add_argument('--num_layers', type=int, default=1, help='num_layers')
    parser.add_argument('--output_size', type=int, default=1, help='output_size')
    parser.add_argument('--bidirectional', type=bool, default=False, help='use bidirectional or not')
    parser.add_argument('--qkv', type=int, default=5, help='dimension for query, key and value')

    parser.add_argument('--which_data', type=str, default='./data/energydata_complete.csv', help='which data to use')
    parser.add_argument('--weights_path', type=str, default='./results/weights/', help='weights path')
    parser.add_argument('--plots_path', type=str, default='./results/plots/', help='plots path')
    parser.add_argument('--numpy_path', type=str, default='./results/numpy/', help='numpy path')

    parser.add_argument('--train_split', type=float, default=0.8, help='train_split')
    parser.add_argument('--test_split', type=float, default=0.5, help='test_split')

    parser.add_argument('--time_plot', type=int, default=100, help='time stamp for plotting')
    parser.add_argument('--num_epochs', type=int, default=500, help='total epoch')
    parser.add_argument('--print_every', type=int, default=10, help='print statistics for every default epoch')

    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--lr_scheduler', type=str, default='cosine', help='learning rate scheduler', choices=['step', 'plateau', 'cosine'])

    config = parser.parse_args()

    torch.cuda.empty_cache()
    main(config)