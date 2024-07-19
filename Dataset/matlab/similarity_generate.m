% compute jaccard similarity

Nets = {'mat_DO_Metabolite','mat_DO_circRNA', 'mat_Lnc_mRNA', 'mat_Lnc_Protein', 'mat_Lnc_RBP'};
	

for i = 1 : length(Nets)
	tic
	inputID = char(strcat('../data/', Nets(i), '.txt'));
	M = load(inputID);
	Sim = 1 - pdist(M, 'jaccard');
	Sim = squareform(Sim);

	Sim = Sim + eye(size(M,1));
	Sim(isnan(Sim)) = 0;      
    [m,n]=size(Sim);
	outputID = char(strcat('../network/Sim_', Nets(i), '.txt'));
	dlmwrite(outputID, Sim, '\t');
	toc
end

% compute gaussian interaction profile kernels similarity
LD = load('../data/mat_Lnc_DO.txt');
association = cell(11311,2);
[ lncRNA_gs ] = GIP( LD );
dlmwrite('../network/Sim_mat_Lnc_Lnc.txt',  lncRNA_gs, '\t');


M = load('../data/Similarity_Matrix_DO.txt');
dlmwrite('../network/Sim_mat_DO_DO.txt',  M, '\t');

