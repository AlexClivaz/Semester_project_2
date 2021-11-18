head = referenceEllipsoid;
head.Name = 'head';
head.LengthUnit = 'centimeter';
head.SemimajorAxis = 19.7/2;
head.SemiminorAxis = 14.5/2;

name_points = ["Fp1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",...
    "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz"];
long = [-18, 18, -36, 36, -90, 90, -144, 144, -162, 162,...
    -54, 54, -90, 90, -126, 126, 0, 0, 180];
lat = [18, 18, 36, 36, 54, 54, 36, 36, 18, 18, 18, 18, ...
    18, 18, 18, 18, 54, 90, 54];

points = struct([]);
M = length(name_points);
for i = 1:M
    points(i).name = name_points(i);
    points(i).long = long(i);
    points(i).lat = lat(i);
end
dist = zeros(M);
for i = 1:M
    for j = 1:M
        dist(i,j) = distance(points(i).lat, points(i).long, points(j).lat, points(j).long, head);
    end
end
sqsum = sum(dist.^2, 'all')/M^2;
edge = exp(-dist.^2 / sqsum);
T = array2table(edge, 'VariableNames', name_points, 'RowNames', name_points);
% writematrix(edge, "edge.csv");
imshow(edge, 'InitialMagnification', 1000)