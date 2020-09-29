<?php

$str_json = file_get_contents('php://input'); //($_POST doesn't work here)
$username = posix_getpwuid(posix_geteuid())['name'];

# decode json to answer page (possibly quality control)
$response = json_decode($str_json, true); // decoding received JSON to array

$mapMarkers = $response["map"];
$shot = $response["shot"];

# time stamp (always GMT)
$timestamp = date('Y-m-d_H-i-s'); 

# write json to file
# requires writing hability on the host for www-data user (33)
# easily done by chown www-data positions
$saveLocation = 'positions/'.$timestamp.'_GMT_positions.json';
$fp = fopen($saveLocation, 'w');
fwrite($fp, $str_json);
fclose($fp);


echo 'Well received '.$timestamp.' GMT'; # by'.getmyuid().':'.get_current_user().'.'.$username.' '.$saveLocation.'error';
// <div align="center">
// <h5> Received data: </h5>
// <table border="1" style="border-collapse: collapse;">
//  <tr> <th> First Name</th> <th> Age</th> </tr>
//  <tr>
//  <td> <center> '.$lName.'<center></td>
//  <td> <center> '.$age.'</center></td>
//  </tr>
//  </table></div>
//  ';
?>
