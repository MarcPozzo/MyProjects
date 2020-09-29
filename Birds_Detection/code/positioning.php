<!DOCTYPE html>
<html>
    <head lang="en">
        <meta charset='utf-8'>
        <meta name="viewport" content="width=device-width,height=device-height, user-scalable=no" />
        <title>Editable annotation on picture</title>

        <link rel="stylesheet" href="leaflet/leaflet.css">
        <script src="jquery.min.js"></script>
        <script src="leaflet/leaflet.js"></script>

        <script type="text/javascript" src="Leaflet.Icon.Glyph/Leaflet.Icon.Glyph.js"></script>

        <script src="Leaflet.Editable/src/Leaflet.Editable.js"></script>

        <script src="base.js"></script>
        <link rel="stylesheet" href="base.css">

    </head>
    <?php 
    $flagEconomy = "signalFiles/flagEconomy.txt";
    if(!file_exists($flagEconomy)){
        $economyState = "Economy mode";
    }else{
        // le fichier existe
        $fp = fopen($flagEconomy,"r");
        $economyState = rtrim(fgets($fp));
        fclose($fp);
    }
    
    $flagBoucle = "signalFiles/flagBoucle.txt";
    if(!file_exists($flagBoucle)){
        $boucleState = "Start";
    }else{
        // le fichier existe
        $fp = fopen($flagBoucle,"r");
        $boucleState = rtrim(fgets($fp));
        fclose($fp);
    }
    ?>

<?php
$date = date("d-m-Y");
Print(" Date : $date");
?>    
    
    <body>
        Landmark positionning <a href="mapSelector.php">back</a>
            <button type="submit" name="submit_show" id="submit_show" value="show" onclick="actionSend()">Send</button>
            <span id="result"></span>
            
        Energy : <input type="button" value="<?php echo($economyState)?>" id="economyModeButton" onclick="economyClick()"></input> 

        Surveillance : analysis <input type="button" value="<?php echo($boucleState)?>" id="analMode" onclick="analyseClick()"></input>
        
        observation : <input type="button" value="<?php echo($boucleState)?>" id="obsMode" onclick="observationClick()"></input>
        
        Pi : <input type="button" value="Reboot" id="RebootThePi" onclick='sendToPython("Reboot")'></input> 
         or <input type="button" value="Shutdown" id="ShutdownThePi" onclick='sendToPython("Shutdown")'></input> 
        
        Mise à jour : <input type="button" value="Mettre à jour" id="miseAjour" onclick="miseAjour()"></input>
        
        Mise à l'heure : <input type="button" value="Mettre à l'heure" id="miseAlheure" onclick="miseAlheure()"></input>
        
<?php
$heure = date("H:i");
Print("Heure de connexion : $heure");
?>        
            
        <div id='map'></div>

        <div id='shot'></div>
        
        <script type="text/javascript">
//     Parameters
// // tournesol Agro
// var mapImage = 'tournesolAgro.png';
// var sizeMap = [70.1,146.2]; # exchange x/y and divide by 10 the result of identify myfile
// var shotImage = 'currentImage.jpg';
// var sizeShot = [194.4,259.2];

// piege versailles 1
// var mapImage = 'maps/versaillesPiege1map.png';
// var sizeMap = [93.1,126.8];
// var shotImage = 'photos/versaillesPiege1shot.jpg';
// var sizeShot = [183.2,326.4];

// selected map
<?php 
$imgFile = ltrim($_GET['mapImage']);
$outMapImage = "var mapImage = \"".$imgFile."\";\n";
$zoomFact = 2;
function MakeArrayImage($imgFile,$zoomFact){
    $imgSize = getimagesize($imgFile);
    $imgHeight = $imgSize[1]/$zoomFact;
    $imgWidth = $imgSize[0]/$zoomFact;
    return("[".$imgHeight.",".$imgWidth."]");
}
$outMapImage = $outMapImage."var sizeMap = ".MakeArrayImage($imgFile,$zoomFact).";\n";
echo($outMapImage);
?>
// last photo
// update needs reloading, dig dipper with https://stackoverflow.com/questions/1077041/refresh-image-with-a-new-one-at-the-same-url
var shotImage = 'photos/stillCurrent.jpeg?t=' + new Date().getTime();
var sizeShot = <?php echo(MakeArrayImage('photos/stillCurrent.jpeg',$zoomFact*2));?>;

//     Methods
L.EditControl = L.Control.extend({

    options: {
        position: 'topleft',
        callback: null,
        kind: '',
        html: ''
    },

    onAdd: function (map) {
        var container = L.DomUtil.create('div', 'leaflet-control leaflet-bar'),
        link = L.DomUtil.create('a', '', container);

        link.href = '#';
        link.title = 'Create a new ' + this.options.kind;
        link.innerHTML = this.options.html;
        L.DomEvent.on(link, 'click', L.DomEvent.stop)
            .on(link, 'click', function () {
                window.LAYER = this.options.callback.call(map.editTools);
            }, this);

        return container;
    }
});

MapConstructor = function(mapName,fileImage,imageSize){
    var map = L.map(mapName, {
        crs: L.CRS.Simple,
        editable: true
    });

    var boundsMap = [[0,0], imageSize];
    var imageMap = L.imageOverlay(fileImage,boundsMap).addTo(map);
    map.fitBounds(boundsMap);

    map.markerCounter = 1; // the counter for the points
    L.NewMarkerControl = L.EditControl.extend({

        options: {
            position: 'topleft',
            callback: (function(latlng,options){
                var icon = L.icon.glyph({ prefix: '', cssClass:'xolonium', glyph: map.markerCounter });
                map.markerCounter = map.markerCounter + 1;
                if (options === undefined) options = {icon: icon};
                else options.icon = icon;

                out = map.editTools.startMarker(latlng,options);
                return(out)
            }),
            kind: 'marker',
            html: '🖈'
        }
    });


    map.addControl(new L.NewMarkerControl());

    // delete shape possible
    var deleteShape = function (e) {
        if ((e.originalEvent.ctrlKey || e.originalEvent.metaKey) && this.editEnabled()) this.editor.deleteShapeAt(e.latlng);
    };
    var deleteMarker = function (e) {
        if ((e.originalEvent.ctrlKey || e.originalEvent.metaKey) && this.editEnabled()) this.remove();
    };

    map.on('layeradd', function (e) {
        if (e.layer instanceof L.Path){
            e.layer.on('click', L.DomEvent.stop).on('click', deleteShape, e.layer);
            e.layer.on('dblclick', L.DomEvent.stop).on('dblclick', e.layer.toggleEdit);
        }else if (e.layer instanceof L.Marker) {
            e.layer.on('click', L.DomEvent.stop).on('click', deleteMarker, e.layer);
            // make numbered markers
        }
    });
    // function to get the markers and polygons to send
    map.GetJSONMarkers = function(){
        var markers = [];
        var polygons = [];
        this.eachLayer( function(layer) {
            if(!(layer instanceof L.Path) && (layer instanceof L.Marker)) {
                if(layer._icon.innerText != ""){
                    markers.push({id: layer._icon.innerText,coord: layer.getLatLng()});
                }
            }else if (layer instanceof L.Path){
                polygons.push(layer.toGeoJSON());
            }

        });
        features = {markers,polygons};
        return features;
    }

    return(map)
}

// set up the map panel
var map = MapConstructor('map',mapImage,sizeMap);
L.CameraPositionControl = L.EditControl.extend({

    options: {
        position: 'topleft',
        callback: (function(latlng,options){
            var icon = L.icon.glyph({ prefix: '', cssClass:'xolonium', glyph: 'C' });
            if (options === undefined) options = {icon: icon};
            else options.icon = icon;

            out = map.editTools.startMarker(latlng,options);
            return(out)
        }),
        kind: 'camera',
        html: 'C'
    }
});
map.addControl(new L.CameraPositionControl());

// set up the photo panel
var shot = MapConstructor('shot',shotImage,sizeShot);
L.NewPolygonControl = L.EditControl.extend({

    options: {
        position: 'topleft',
        callback: shot.editTools.startPolygon,
        kind: 'polygon',
        html: '▰'
    }
});
shot.addControl(new L.NewPolygonControl());

// set up the sending
function actionSend() {
    if (window.XMLHttpRequest) {// code for IE7+, Firefox, Chrome, Opera, Safari
        xmlhttp = new XMLHttpRequest();
    }
    else {// code for IE6, IE5
        xmlhttp = new ActiveXObject("Microsoft.XMLHTTP");
    }
    // get time stamp
    var d = new Date(0); // The 0 there is the key, which sets the date to the epoch
        d.setUTCSeconds(Date.now()/1000);
    var values = {
        time: d,
        mapFile: mapImage,
        sizeMap: sizeMap,
        latlong: parse_query_string(window.location.search.substring(1)),
        shotFile: shotImage,
        sizeShot: sizeShot,
        map:map.GetJSONMarkers(),
        shot:shot.GetJSONMarkers()};
    var myJsonString = JSON.stringify(values);
    xmlhttp.onreadystatechange = respond;
    xmlhttp.open("POST", "ajax-save.php", true);
    xmlhttp.send(myJsonString);
}
// receiving response after sending
function respond() {
    if (xmlhttp.readyState == 4 && xmlhttp.status == 200) {
        document.getElementById('result').innerHTML = xmlhttp.responseText;
    }
}
// get the parameters (lat,long)
function parse_query_string(query) {
    var vars = query.split("&");
    var query_string = {};
    for (var i = 0; i < vars.length; i++) {
        var pair = vars[i].split("=");
        var key = decodeURIComponent(pair[0]);
        var value = decodeURIComponent(pair[1]);
        // If first entry with this name
        if (typeof query_string[key] === "undefined") {
            query_string[key] = decodeURIComponent(value);
            // If second entry with this name
        } else if (typeof query_string[key] === "string") {
            var arr = [query_string[key], decodeURIComponent(value)];
            query_string[key] = arr;
            // If third or later entry with this name
        } else {
            query_string[key].push(decodeURIComponent(value));
        }
    }
    return query_string;
}

// set up the sending of commands to python
function sendToPython(value,response=respond) {
    if (window.XMLHttpRequest) {// code for IE7+, Firefox, Chrome, Opera, Safari
        xmlhttp = new XMLHttpRequest();
    }
    else {// code for IE6, IE5
        xmlhttp = new ActiveXObject("Microsoft.XMLHTTP");
    }
    // get time stamp
    var values = {"cmd":value};
    var myJsonString = JSON.stringify(values);
    // xmlhttp.onreadystatechange = response;
    xmlhttp.open("POST", location.protocol+"//"+window.location.hostname+":8082", response);
    xmlhttp.send(myJsonString);
}

function economyClick(){
    
    if(document.getElementById("economyModeButton").value == "Economy mode"){
        sendToPython("energySavings");
        document.getElementById("economyModeButton").value = "Normal mode";
    }
    else{
        sendToPython("noEnergySavings");
        document.getElementById("economyModeButton").value = "Economy mode";
    }
}

function update(filename,id) {
  fetch(filename).then((resp) => resp.text()).then(function(data) {
    document.getElementById(id).value = data;
    return data;
  });
}

function updateUntilChange(filename,id1,id2) {
  fetch(filename).then((resp) => resp.text()).then(function(data) {
      init = document.getElementById(id1).value;
      if(init == data){
          setTimeout(function(){
               updateUntilChange(filename,id1,id2)
          }, 5000);
      }else{
          document.getElementById(id1).value = data;
          document.getElementById(id1).disabled = false;
          document.getElementById(id2).disabled = false;
      }
    return data;
  });
}

function analyseClick(){
    
    if(document.getElementById("analMode").value == "Start"){
        sendToPython("lancerBoucleAnal");
        document.getElementById("analMode").value = "Stop";
        document.getElementById("obsMode").disabled = true;
	// should grey out obsMode
    }else{ // ask to stop
        sendToPython("stopBoucle");
        // document.getElementById("analMode").value = "Waiting...";
        document.getElementById("analMode").disabled = true;
	updateUntilChange("<?php echo($flagBoucle)?>","analMode","obsMode");

    }

}
function observationClick(){
    
    if(document.getElementById("obsMode").value == "Start"){
        sendToPython("lancerBoucleObs");
        document.getElementById("obsMode").value = "Stop";
        document.getElementById("analMode").disabled = true;
    // should grey out analMode
    }else{
        sendToPython("stopBoucle");
        document.getElementById("obsMode").disabled = true;
	updateUntilChange("<?php echo($flagBoucle)?>","obsMode","analMode");
    }
}
function miseAjour(){
    
    if (document.getElementById("miseAjour").value == "Mettre à jour"){
        sendToPython("miseAjour");
        document.getElementById("miseAjour").value = "Teminé";
    }else{
        sendToPython("git pull");
        document.getElementById("miseAjour").value = "Terminé";    
    }

}
function miseAlheure(){
    sendToPython("MettreAlheure");

}
        
            
        </script>

    </body>
</html>

