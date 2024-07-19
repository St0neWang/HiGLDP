function [ lncRNASim ] = GIP( adjMat )

nm=size(adjMat,1);
normSum=0;
for i=1:nm
    
   normSum=normSum+ ((norm(adjMat(i,:),2)).^2);
    
end

rm=1/(normSum/nm);

lncRNASim = zeros(nm,nm);

for i=1:nm
   for j=1:nm
       sub=adjMat(i,:)-adjMat(j,:);
       lncRNASim(i,j)=exp(-rm*((norm(sub,2)).^2));
       
   end 
    
end

end

