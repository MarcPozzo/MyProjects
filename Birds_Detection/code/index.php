<?php 
$serverURL = "http://$_SERVER[HTTP_HOST]";
shell_exec("curl -d '{\"cmd\":\"startMotion\"}' ".$_SERVER[HTTP_HOST].":8082"); sleep(1); 
?>
<!DOCTYPE html>
<html>
    <head lang="en">
        <meta charset='utf-8'>
        <meta name="viewport" content="width=device-width,height=device-height, user-scalable=no" />
        <title>Choice of angle</title>

        <script src="base.js"></script>
        <link rel="stylesheet" href="base.css">

    </head>
    <body>
        <h2>C3PO: adjust your view</h2>

        <div id='liveStream'>
        <iframe src=<?php echo("'".$serverURL.":8081'");?> align="center" width="640" height="480" scrolling="no" frameborder=no marginheight="0px"></iframe>
        <button onclick='sendToPython("startMotion")'>Start motion</button> 
        <button onclick='sendToPython("stopMotion")'>Stop motion</button> 
        </div>


        <!-- risk to need https settings for it to work fine in real life-->
        <button onclick="getLocation()">Get location</button> 
        <p id="demo"></p>


        <button type="submit" name="submit_show" id="submit_show" value="show" onclick="actionSend()">Done
        </button>
        <span id="result"></span>

        <script>
// geolocation scripts
var x = document.getElementById("demo");

var positionG = {};
positionG.coords = {latitude:0,longitude:0};
function getLocation() {
    if (navigator.geolocation) {
        navigator.geolocation.getCurrentPosition(showPosition);
    } else { 
        x.innerHTML = "Geolocation is not supported by this browser.";
    }
}

function showPosition(position) {
    positionG = position;
    x.innerHTML = "Latitude: " + position.coords.latitude + 
        "<br>Longitude: " + position.coords.longitude;
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
// receiving response after sending
function respond() {
    if (xmlhttp.readyState == 4 && xmlhttp.status == 200) {
        document.getElementById('result').innerHTML = xmlhttp.responseText;
    }
}

// send and next step
function actionSend(){
    var toSend = {positionG:positionG};
    // var out = sendToPython("stopMotion",function(){
    var out = sendToPython("savePicture",false);
    // });
    var options = 'lat='+positionG.coords.latitude+'&long='+positionG.coords.longitude;
    window.location.replace("mapSelector.php?"+options);
}

        </script>
    </body>
</html>

