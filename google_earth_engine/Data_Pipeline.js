// define your var geometry on Google Earth Engine's map

//constants determining downloaded image size/resolution
var R_e = 6370;
var alfa = (1/R_e) * 180/(3.1415);
var gsd = 250; //meters per pixel
var box_size = gsd/1000 * 400; //pixels per image (roughly)

//create joint dataset with all the necessary bands, 5% cloud cover
var proc_collection = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
    .filterBounds(geometry)
    .filterDate('2017-01-01', '2022-01-01')
    .select(['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'ST_B10'])
    .filter('CLOUD_COVER < .05');

var raw_collection = ee.ImageCollection('LANDSAT/LC08/C02/T1')
    .filterBounds(geometry)
    .filterDate('2017-01-01', '2022-01-01')
    .select(['B1', 'B2', 'B3', 'B4', 'B10'])
    .filter('CLOUD_COVER < .05');

var filter = ee.Filter.equals({
  leftField: 'system:index',
  rightField: 'system:index'
});

var innerJoin = ee.Join.inner('primary', 'secondary');
var toyJoin = innerJoin.apply(proc_collection, raw_collection, filter);
var joined = toyJoin.map(function(feature) {
  return ee.Image.cat(feature.get('primary'), feature.get('secondary'));
});
print('Dataset:', joined);
var n = joined.size().getInfo();
var colList = joined.toList(n);

//crop and export each image
for(var i = 0; i < n; i++){
    var img = ee.Image(colList.get(i));
    var name = img.get('system:index').getInfo().substring(0,20);
    Map.centerObject(img);
    Map.addLayer(img, {bands: ['B1']}, 'aerosol');
    var center = Map.getCenter().coordinates().getInfo();

    var one = ee.Number(center[0]).add(ee.Number(-box_size * alfa/2));
    var two = ee.Number(center[1]).add(ee.Number(-box_size * alfa/2));
    var three = ee.Number(center[0]).add(ee.Number(box_size * alfa/2));
    var four = ee.Number(center[1]).add(ee.Number(box_size * alfa/2));
    var site = ee.Geometry.Rectangle([one, two, three, four])

    Export.image.toDrive({
      image: img,
      scale: gsd,
      description: name,
      region: site,
      folder: 'pacific',
      fileFormat: 'GeoTIFF',
      formatOptions: {"cloudOptimized": false}
      });
}