ptm = Sys.time()
library(CellChat)
library(patchwork)
library(glue)


scPalette_JQ <- function(n) {
  colorSpace <- c("#1f77b4","#aec7e8","#ff7f0e","#ffbb78","#2ca02c","#98df8a","#d62728","#ff9896",
                  "#9467bd","#c5b0d5","#8c564b","#c49c94","#e377c2","#f7b6d2","#7f7f7f","#c7c7c7",
                  "#bcbd22","#dbdb8d","#17becf","#9edae5")
  if (n <= length(colorSpace)) {
    colors <- colorSpace[1:n]
  } else {
    colors <- grDevices::colorRampPalette(colorSpace)(n)
  }
  return(colors)
}

top=1
weight.scale = FALSE
edge.width.max=20000



data.dir <- '/zebrafish-circle'
dir.create(data.dir)
setwd(data.dir)

for (i in (0:11)){
  
  ## result from function Derive_TypeAtt()
  file_path = glue("/Zebrafish/Zebrafish_{i}.csv")
  title.name = glue("t={i}")
  data <- read.csv(file_path, header = TRUE, row.names = 1)
  data_matrix <- as.matrix(data)
  A <- t(data_matrix)
  
  M_names <- c('blastomeres', 'periderm', 'axial', 'neural', 'muscle', 'hematopoietic', 'endoderm', 'pgc', 'eye', 'cephalic', 'pronephros', 'glial')
  M <- matrix(0, nrow = 12, ncol = 12, dimnames = list(M_names, M_names))
  row_indices <- match(rownames(A), rownames(M))
  col_indices <- match(colnames(A), colnames(M))
  M[row_indices, col_indices] <- A
  
  
  net <- M
  thresh <- stats::quantile(net, probs = 1-top)
  net[net < thresh] <- 0
  
  
  g <- graph_from_adjacency_matrix(net, mode = "directed", weighted = T)
  edge.start <- igraph::ends(g, es=igraph::E(g), names=FALSE)
  layout=in_circle()
  coords<-layout_(g,layout)
  coords_scale=scale(coords)
  color.use = scPalette_JQ(length(igraph::V(g)))
  vertex.weight = 20
  vertex.weight.max <- max(vertex.weight)
  vertex.size.max <- 5
  vertex.weight <- vertex.weight/vertex.weight.max*vertex.size.max+5
  vertex.label.color= "black"
  vertex.label.cex=1
  loop.angle<-ifelse(coords_scale[igraph::V(g),1]>0,-atan(coords_scale[igraph::V(g),2]/coords_scale[igraph::V(g),1]),pi-atan(coords_scale[igraph::V(g),2]/coords_scale[igraph::V(g),1]))
  igraph::V(g)$size<-vertex.weight
  igraph::V(g)$color<-color.use[igraph::V(g)]
  igraph::V(g)$frame.color <- color.use[igraph::V(g)]
  igraph::V(g)$label.color <- vertex.label.color
  igraph::V(g)$label.cex<-vertex.label.cex
  
  
  
  
  
  edge.weight.max <- max(igraph::E(g)$weight)
  if (weight.scale == TRUE) {
    igraph::E(g)$width<- igraph::E(g)$weight/edge.weight.max*edge.width.max
  }else{
    igraph::E(g)$width<-edge.width.max*igraph::E(g)$weight
  }
  
  arrow.width=1
  arrow.size = 0.5
  edge.label.color='black'
  edge.label.cex=1
  alpha.edge = 0.8
  igraph::E(g)$arrow.width<-arrow.width
  igraph::E(g)$arrow.size<-arrow.size
  igraph::E(g)$label.color<-edge.label.color
  igraph::E(g)$label.cex<-edge.label.cex
  igraph::E(g)$color<- grDevices::adjustcolor(igraph::V(g)$color[edge.start[,1]],alpha.edge)
  igraph::E(g)$loop.angle <- rep(0, length(igraph::E(g)))
  
  if(sum(edge.start[,2]==edge.start[,1])!=0){
    igraph::E(g)$loop.angle[which(edge.start[,2]==edge.start[,1])]<-loop.angle[edge.start[which(edge.start[,2]==edge.start[,1]),1]]
  }
  
  radian.rescale <- function(x, start=0, direction=1) {
    c.rotate <- function(x) (x + start) %% (2 * pi) * direction
    c.rotate(scales::rescale(x, c(0, 2 * pi), range(x)))
  }
  label.locs <- radian.rescale(x=1:length(igraph::V(g)), direction=-1, start=0)
  label.dist <- vertex.weight/max(vertex.weight)+2
  
  edge.curved=0.2
  shape='circle'
  margin=0.2
  
  png(glue("zebrafish-circle/t={i}.png"), width = 800, height = 800, res = 150)
  
  
  plot(g,edge.curved=edge.curved,vertex.shape=shape,layout=coords_scale,margin=margin, vertex.label.dist=label.dist,
       vertex.label.degree=label.locs, vertex.label.family="Helvetica", edge.label.family="Helvetica") # "sans"
  
  title(title.name, cex.main = 3)
  
  
  dev.off()
  
}
