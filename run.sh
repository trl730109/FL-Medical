python experiments_new.py --model=resnet50 \
	--dataset=liver \
	--alg=fedavg \
	--lr=0.0001 \
	--batch-size=16 \
	--epochs=1 \
	--n_parties=4 \
	--rho=0.9 \
	--comm_round=10 \
	--partition=noniid-labeldir \
	--beta=0.5 \
	--device="cuda:2" \
	--datadir='./data/' \
	--optimizer='Adam' \
	--logdir='./logs/' \
	--model_weight_path='/mnt/raid/tangzichen/Liver/resnet50-pre.pth' \
	--noise=0 \
	--init_seed=0