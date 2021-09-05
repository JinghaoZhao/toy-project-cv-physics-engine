const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera( 75, window.innerWidth / window.innerHeight, 0.1, 1000 );

const renderer = new THREE.WebGLRenderer();
renderer.setSize( window.innerWidth, window.innerHeight );
document.body.appendChild( renderer.domElement );

const geometry = new THREE.BoxGeometry();
const material = new THREE.MeshBasicMaterial( { color: 0x00ff00 } );
const cube = new THREE.Mesh( geometry, material );
scene.add( cube );

camera.position.z = 5;

function animate() {
    requestAnimationFrame( animate );
    //console.log(cube.position.x);
	renderer.render( scene, camera );
}
animate();

var changeCoord = function() {
    camera.position.x = document.getElementById("xVal").value;
    camera.position.y = document.getElementById("yVal").value;
    camera.position.z = document.getElementById("zVal").value;
}

var socket = io();
socket.on("update", function(msg) {
    cube.position.x = parseFloat(msg);
    //console.log(msg);
});

let count = 0;
socket.on("multicast", function(msg) {
    count++;
    console.log(count);
});