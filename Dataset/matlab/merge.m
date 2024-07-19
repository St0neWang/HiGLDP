function [Q] = merge(networks, rsp, maxiter)
	Q = [];
	for i = 1 : length(networks)
		fileID = char(strcat('../network/', networks(i), '.txt'));
		net = load(fileID);
		tQ = RWR(net, maxiter, rsp);
		Q = [Q, tQ];
    end
    nnode = size(Q, 1);
	alpha = 1 / nnode;
	Q = log(Q + alpha) - log(alpha);
end
