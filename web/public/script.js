/***
 * A demo page for loading glTF models
 */

import * as THREE from 'https://threejsfundamentals.org/threejs/resources/threejs/r122/build/three.module.js';
import {OrbitControls} from 'https://threejsfundamentals.org/threejs/resources/threejs/r122/examples/jsm/controls/OrbitControls.js';
import {GLTFLoader} from 'https://threejsfundamentals.org/threejs/resources/threejs/r122/examples/jsm/loaders/GLTFLoader.js';

function main() {
    var socket = io();

    socket.on("multicast", function(msg) {
        let message = JSON.parse(msg);
        if (startTime < 0) {
            startTime = message.shift().time;
        } else {
            message.shift();
        }
        console.log(message);

        for (const element of message) {
            // console.log(element.name);
            let name = element.name.split(".");
            if (!(name[0] in data3D)) {
                continue;
            }
            for (const innerEle in element) {
                if (innerEle != 'name') {
                    try {
                        data3D[name[0]][name[1]].push(innerEle);
                        data3D[name[0]][name[1]].push(element[innerEle]);
                        /*
                        if (startTime < 0) {
                            startTime = Math.ceil(Date.now() - Number(innerEle) * 1000);
                        }
                        */
                    } catch(error) {
                        console.log(element);
                    }
                }
            }
        }
    });


    const canvas = document.querySelector('#c');
    const renderer = new THREE.WebGLRenderer({canvas});
    renderer.shadowMap.enabled = true;
    //renderer.autoClear = true; //USELESS

    const fov = 45;
    const aspect = 2;  // the canvas default
    const near = 0.1;
    const far = 200;
    const camera = new THREE.PerspectiveCamera(fov, aspect, near, far);
    camera.position.set(0, 30, 80);

    const controls = new OrbitControls(camera, canvas);
    controls.target.set(0, 15, 0);
    controls.update();

    const scene = new THREE.Scene();
    scene.background = new THREE.Color('#DEFEFF');

    const list = ['Drone_Controller','Turbine_Controller','Turbine_R','Turbine_L',
    'U_MassPoint','Eye_Controller','Eye_Pupil','D_MassPoint','Drone_Body',
    'Drone_Gen_R','Drone_Panel_R','Drone_leg_R','R_P1_G','R_P2','R_P3_G','R_P4',
    'R_P5_M','R_P6_G','R_P7','Drone_Gen_L','Drone_Panel_L','Drone_leg_L','L_P1_G',
    'L_P2','L_P3_G','L_P4','L_P5_M','L_P6_G','L_P7','Drone_UPanel_R','Drone_UPanel_L',
    'Drone_UPart','Drone_Turb_M_L','Drone_Turb_Blade_L','Drone_Turb_M_R',
    'Drone_Turb_Blade_R','Drone_leg_F','F_P1_G','F_P2','F_P3_G','F_P4','F_P5_M',
    'F_P6_G','F_P7','Drone_ILens','Drone_IEye'];

    var all3d = {};
    // 传入的数据放进data3D里面去，格式稍微修改即可
    var data3D = {
        Drone_Controller: {
            position: [],
            quaternion: [],
            scale: []
        },
        Turbine_Controller: {
            position: [],
            quaternion: [],
            scale: []
        },
        Turbine_R: {
            position: [],
            quaternion: [],
            scale: []
        },
        Turbine_L: {
            position: [],
            quaternion: [],
            scale: []
        },
        U_MassPoint: {
            position: [],
            quaternion: [],
            scale: []
        },
        Eye_Controller: {
            position: [],
            quaternion: [],
            scale: []
        },
        Eye_Pupil: {
            position: [],
            quaternion: [],
            scale: []
        },
        D_MassPoint: {
            position: [],
            quaternion: [],
            scale: []
        },
        Drone_Body: {
            position: [],
            quaternion: [],
            scale: []
        },
        Drone_Gen_R: {
            position: [],
            quaternion: [],
            scale: []
        },
        Drone_Panel_R: {
            position: [],
            quaternion: [],
            scale: []
        },
        Drone_Leg_R: {
            position: [],
            quaternion: [],
            scale: []
        },
        R_P1_G: {
            position: [],
            quaternion: [],
            scale: []
        },
        R_P2: {
            position: [],
            quaternion: [],
            scale: []
        },
        R_P3_G: {
            position: [],
            quaternion: [],
            scale: []
        },
        R_P4: {
            position: [],
            quaternion: [],
            scale: []
        },
        R_P5_M: {
            position: [],
            quaternion: [],
            scale: []
        },
        R_P6_G: {
            position: [],
            quaternion: [],
            scale: []
        },
        R_P7: {
            position: [],
            quaternion: [],
            scale: []
        },
        Drone_Gen_L: {
            position: [],
            quaternion: [],
            scale: []
        },
        Drone_Panel_L: {
            position: [],
            quaternion: [],
            scale: []
        },
        Drone_Leg_L: {
            position: [],
            quaternion: [],
            scale: []
        },
        L_P1_G: {
            position: [],
            quaternion: [],
            scale: []
        },
        L_P2: {
            position: [],
            quaternion: [],
            scale: []
        },
        L_P3_G: {
            position: [],
            quaternion: [],
            scale: []
        },
        L_P4: {
            position: [],
            quaternion: [],
            scale: []
        },
        L_P5_M: {
            position: [],
            quaternion: [],
            scale: []
        },
        L_P6_G: {
            position: [],
            quaternion: [],
            scale: []
        },
        L_P7: {
            position: [],
            quaternion: [],
            scale: []
        },
        Drone_UPanel_R: {
            position: [],
            quaternion: [],
            scale: []
        },
        Drone_UPanel_L: {
            position: [],
            quaternion: [],
            scale: []
        },
        Drone_UPart: {
            position: [],
            quaternion: [],
            scale: []
        },
        Drone_Turb_M_L: {
            position: [],
            quaternion: [],
            scale: []
        },
        Drone_Turb_Blade_L: {
            position: [],
            quaternion: [],
            scale: []
        },
        Drone_Turb_M_R: {
            position: [],
            quaternion: [],
            scale: []
        },
        Drone_Turb_Blade_R: {
            position: [],
            quaternion: [],
            scale: []
        },
        Drone_leg_F: {
            position: [],
            quaternion: [],
            scale: []
        },
        F_P1_G: {
            position: [],
            quaternion: [],
            scale: []
        },
        F_P2: {
            position: [],
            quaternion: [],
            scale: []
        },
        F_P3_G: {
            position: [],
            quaternion: [],
            scale: []
        },
        F_P4: {
            position: [],
            quaternion: [],
            scale: []
        },
        F_P5_M: {
            position: [],
            quaternion: [],
            scale: []
        },
        F_P6_G: {
            position: [],
            quaternion: [],
            scale: []
        },
        F_P7: {
            position: [],
            quaternion: [],
            scale: []
        },
        Drone_ILens: {
            position: [],
            quaternion: [],
            scale: []
        },
        Drone_IEye: {
            position: [],
            quaternion: [],
            scale: []
        }
    };



    {
        const skyColor = 0xB1E1FF;  // light blue
        const groundColor = 0xB97A20;  // brownish orange
        const intensity = 1;
        const light = new THREE.HemisphereLight(skyColor, groundColor, intensity);
        scene.add(light);
    }

    {
        const color = 0xFFFFFF;
        const intensity = 1;
        const light = new THREE.DirectionalLight(color, intensity);
        light.castShadow = true;
        light.position.set(-250, 800, 850);
        light.target.position.set(-550, 40, -450);

        light.shadow.bias = -0.004;
        light.shadow.mapSize.width = 2048;
        light.shadow.mapSize.height = 2048;

        scene.add(light);
        scene.add(light.target);
        const cam = light.shadow.camera;
        cam.near = 1;
        cam.far = 2000;
        cam.left = -1500;
        cam.right = 1500;
        cam.top = 1500;
        cam.bottom = -1500;

        const cameraHelper = new THREE.CameraHelper(cam);
        scene.add(cameraHelper);
        cameraHelper.visible = false;
        const helper = new THREE.DirectionalLightHelper(light, 100);
        scene.add(helper);
        helper.visible = false;

        function updateCamera() {
            // update the light target's matrixWorld because it's needed by the helper
            light.updateMatrixWorld();
            light.target.updateMatrixWorld();
            helper.update();
            // update the light's shadow camera's projection matrix
            light.shadow.camera.updateProjectionMatrix();
            // and now update the camera helper we're using to show the light's shadow camera
            cameraHelper.update();
        }

        updateCamera();
    }

    function resizeRendererToDisplaySize(renderer) {
        const canvas = renderer.domElement;
        const width = canvas.clientWidth;
        const height = canvas.clientHeight;
        const needResize = canvas.width !== width || canvas.height !== height;
        if (needResize) {
            renderer.setSize(width, height, false);
        }
        return needResize;
    }


    function dumpObject(obj, lines = [], isLast = true, prefix = '') {
        const localPrefix = isLast ? '└─' : '├─';
        lines.push(`${prefix}${prefix ? localPrefix : ''}${obj.name || '*no-name*'} [${obj.type}]`);
        const newPrefix = prefix + (isLast ? '  ' : '│ ');
        const lastNdx = obj.children.length - 1;
        obj.children.forEach((child, ndx) => {
            const isLast = ndx === lastNdx;
            dumpObject(child, lines, isLast, newPrefix);
        });
        return lines;
    }

    function frameArea(sizeToFitOnScreen, boxSize, boxCenter, camera) {
        const halfSizeToFitOnScreen = sizeToFitOnScreen * 0.5;
        const halfFovY = THREE.MathUtils.degToRad(camera.fov * .5);
        const distance = halfSizeToFitOnScreen / Math.tan(halfFovY);
        // compute a unit vector that points in the direction the camera is now
        // in the xz plane from the center of the box
        const direction = (new THREE.Vector3())
            .subVectors(camera.position, boxCenter)
            .multiply(new THREE.Vector3(1, 0, 1))
            .normalize();

        // move the camera to a position distance units way from the center
        // in whatever direction the camera was from the center already
        camera.position.copy(direction.multiplyScalar(distance).add(boxCenter));

        // pick some near and far values for the frustum that
        // will contain the box.
        camera.near = boxSize / 100;
        camera.far = boxSize * 100;

        camera.updateProjectionMatrix();

        // point the camera to look at the center of the box
        camera.lookAt(boxCenter.x, boxCenter.y, boxCenter.z);
    }


    var loadedPart;

    function addPart(part) {
        // console.log(part);
        var partChildren = part.children.slice();
        const len = partChildren.length;
        for (let i = 0; i < len; ++i) {
            if (partChildren[i].type === "Object3D")
            all3d[partChildren[i].name] = partChildren[i];
        }
        partChildren.forEach((child, i) => {
            addPart(child);
        });
    }

    {
        const gltfLoader = new GLTFLoader();
        gltfLoader.load('./drone/scene.gltf', (gltf) => {
            // console.log(dumpObject(gltf.scene).join('\n'));
            const root = gltf.scene;
            scene.add(root);
            root.traverse((obj) => {
                if (obj.castShadow !== undefined) {
                    obj.castShadow = true;
                    obj.receiveShadow = true;
                }
            });
            //loadedPart = root.getObjectByName('Drone_Controller');
            loadedPart = root.children[0];
            all3d['Drone_Controller'] = loadedPart;
            root.updateMatrixWorld();
            addPart(loadedPart);
            // console.log(all3d);
            const box = new THREE.Box3().setFromObject(root);
            const boxSize = box.getSize(new THREE.Vector3()).length();
            const boxCenter = box.getCenter(new THREE.Vector3());
            frameArea(boxSize * 0.5, boxSize, boxCenter, camera);
            controls.maxDistance = boxSize * 10;
            controls.target.copy(boxCenter);
            controls.update();
        });
    }

    var startTime = -1;

    function render() {
        let time = startTime < 0 ? -1 : (Date.now() - startTime) / 1000;
        //console.log(time);
        if (resizeRendererToDisplaySize(renderer)) {
            const canvas = renderer.domElement;
            camera.aspect = canvas.clientWidth / canvas.clientHeight;
            camera.updateProjectionMatrix();
        }
        // 更新drone各部分的三个属性
        var arg;
        for (arg in data3D) {
            try {
                if (data3D[arg].position && Number(data3D[arg].position[0]) < time) {
                    var argPosition = data3D[arg].position.shift();
                    argPosition = data3D[arg].position.shift();
                    all3d[arg].position.set(...argPosition);
                }    
                if (data3D[arg].quaternion && Number(data3D[arg].quaternion[0]) < time) {
                    var argQuaternion = data3D[arg].quaternion.shift();
                    argQuaternion = data3D[arg].quaternion.shift();
                    all3d[arg].quaternion.set(...argQuaternion);
                }
                if (data3D[arg].scale && Number(data3D[arg].scale[0]) < time) {
                    var argScale = data3D[arg].scale.shift();
                    argScale = data3D[arg].scale.shift();
                    all3d[arg].scale.set(...argScale);
                }
            } catch(error) {
                console.log(error);
                console.log('-----', arg);
            }
        }

        renderer.render(scene, camera);
        requestAnimationFrame(render);
    }
    requestAnimationFrame(render);

}

main();