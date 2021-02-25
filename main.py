from models.NetMF import netmf
dataset="Flicker"
evlation="node_classfication"
netmf('data/Flickr/Flickr_SDM.mat',evlation,variable_name="Network")
