from IPython.display import HTML, display
import json

class RealisticHumanAvatar:
    def __init__(self):
        self.current_gesture = "neutral"
        
    def _get_html(self):
        """Generate HTML/CSS/JS for realistic human avatar"""
        return """
<!DOCTYPE html>
<html>
<head>
    <style>
        body { margin: 0; padding: 0; overflow: hidden; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
        #container { width: 100%; height: 600px; background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); }
        #controls {
            position: absolute;
            top: 20px;
            left: 20px;
            background: rgba(0,0,0,0.8);
            padding: 20px;
            border-radius: 12px;
            color: white;
            z-index: 100;
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        }
        button {
            margin: 5px;
            padding: 12px 20px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-weight: bold;
            font-size: 13px;
            transition: transform 0.2s;
        }
        button:hover { transform: scale(1.05); }
        .gesture-btn { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); }
        select {
            padding: 10px;
            margin: 8px 0;
            border-radius: 8px;
            border: none;
            font-size: 14px;
            width: 100%;
            background: white;
        }
        h3 { margin-top: 0; font-size: 18px; border-bottom: 2px solid #667eea; padding-bottom: 10px; }
    </style>
</head>
<body>
    <div id="container"></div>
    <div id="controls">
        <h3>🎭 Realistic Avatar</h3>
        <label style="font-size: 13px;">Choose Expression:</label>
        <select id="gestureSelect" onchange="changeGesture(this.value)">
            <option value="neutral">😐 Neutral</option>
            <option value="happy">😊 Happy</option>
            <option value="thinking">🤔 Thinking</option>
            <option value="excited">🤩 Excited</option>
            <option value="sad">😢 Sad</option>
            <option value="surprised">😲 Surprised</option>
            <option value="waving">👋 Waving</option>
            <option value="pointing">👉 Pointing</option>
        </select>
        <br>
        <button onclick="startTalking()">🗣️ Start Talking</button>
        <button onclick="stopTalking()">🤐 Stop</button>
        <br>
        <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid #444;">
            <small style="opacity: 0.8;">Quick Actions:</small><br>
            <button class="gesture-btn" onclick="respond('greeting')">Greet</button>
            <button class="gesture-btn" onclick="respond('explaining')">Explain</button>
            <button class="gesture-btn" onclick="respond('thinking')">Think</button>
        </div>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script>
        let scene, camera, renderer, avatar;
        let head, leftEye, rightEye, leftEyebrow, rightEyebrow, mouth, body, leftArm, rightArm;
        let isTalking = false;
        let currentGesture = 'neutral';
        let animationFrame = 0;

        function init() {
            scene = new THREE.Scene();
            scene.background = new THREE.Color(0x1e3c72);
            scene.fog = new THREE.Fog(0x1e3c72, 8, 15);
            
            camera = new THREE.PerspectiveCamera(50, window.innerWidth / 600, 0.1, 1000);
            camera.position.set(0, 1.5, 4);
            camera.lookAt(0, 1.5, 0);
            
            renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
            renderer.setSize(window.innerWidth, 600);
            renderer.shadowMap.enabled = true;
            renderer.shadowMap.type = THREE.PCFSoftShadowMap;
            document.getElementById('container').appendChild(renderer.domElement);
            
            // Advanced lighting for realism
            const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
            scene.add(ambientLight);
            
            const keyLight = new THREE.DirectionalLight(0xffffff, 1);
            keyLight.position.set(5, 8, 5);
            keyLight.castShadow = true;
            keyLight.shadow.mapSize.width = 2048;
            keyLight.shadow.mapSize.height = 2048;
            scene.add(keyLight);
            
            const fillLight = new THREE.PointLight(0x88ccff, 0.4);
            fillLight.position.set(-5, 3, 3);
            scene.add(fillLight);
            
            const rimLight = new THREE.PointLight(0xffaa88, 0.3);
            rimLight.position.set(0, 5, -5);
            scene.add(rimLight);
            
            createRealisticAvatar();
            animate();
        }
        
        function createRealisticAvatar() {
            avatar = new THREE.Group();
            
            // More realistic head shape (slightly oval)
            const headGeometry = new THREE.SphereGeometry(0.85, 32, 32);
            headGeometry.scale(1, 1.1, 0.95);
            const headMaterial = new THREE.MeshPhongMaterial({ 
                color: 0xffc9a3,
                shininess: 15,
                specular: 0x222222
            });
            head = new THREE.Mesh(headGeometry, headMaterial);
            head.castShadow = true;
            head.position.y = 1.8;
            avatar.add(head);
            
            // Realistic eyes with multiple layers
            createEyes();
            
            // Eyebrows
            createEyebrows();
            
            // Nose with nostrils
            createNose();
            
            // Mouth with lips
            createMouth();
            
            // Ears
            createEars();
            
            // Neck
            const neckGeometry = new THREE.CylinderGeometry(0.25, 0.3, 0.4, 16);
            const neckMaterial = new THREE.MeshPhongMaterial({ color: 0xffc9a3 });
            const neck = new THREE.Mesh(neckGeometry, neckMaterial);
            neck.position.y = 1.15;
            avatar.add(neck);
            
            // Realistic body with clothing
            createBody();
            
            // Arms with realistic joints
            createArms();
            
            // Hair
            createHair();
            
            scene.add(avatar);
        }
        
        function createEyes() {
            // Left Eye
            leftEye = new THREE.Group();
            
            // Eye socket (slight depth)
            const socketGeometry = new THREE.SphereGeometry(0.18, 16, 16);
            const socketMaterial = new THREE.MeshPhongMaterial({ color: 0xffa080 });
            const leftSocket = new THREE.Mesh(socketGeometry, socketMaterial);
            leftSocket.scale.z = 0.5;
            leftEye.add(leftSocket);
            
            // Eyeball (white part)
            const eyeballGeometry = new THREE.SphereGeometry(0.15, 16, 16);
            const eyeballMaterial = new THREE.MeshPhongMaterial({ color: 0xffffff, shininess: 50 });
            const leftEyeball = new THREE.Mesh(eyeballGeometry, eyeballMaterial);
            leftEyeball.position.z = 0.02;
            leftEye.add(leftEyeball);
            
            // Iris
            const irisGeometry = new THREE.SphereGeometry(0.08, 16, 16);
            const irisMaterial = new THREE.MeshPhongMaterial({ 
                color: 0x2e5a3d,
                shininess: 80
            });
            const leftIris = new THREE.Mesh(irisGeometry, irisMaterial);
            leftIris.position.z = 0.12;
            leftEye.add(leftIris);
            
            // Pupil
            const pupilGeometry = new THREE.SphereGeometry(0.04, 16, 16);
            const pupilMaterial = new THREE.MeshPhongMaterial({ color: 0x000000, shininess: 100 });
            const leftPupil = new THREE.Mesh(pupilGeometry, pupilMaterial);
            leftPupil.position.z = 0.16;
            leftEye.add(leftPupil);
            
            // Highlight (glint)
            const highlightGeometry = new THREE.SphereGeometry(0.02, 8, 8);
            const highlightMaterial = new THREE.MeshBasicMaterial({ color: 0xffffff });
            const leftHighlight = new THREE.Mesh(highlightGeometry, highlightMaterial);
            leftHighlight.position.set(0.03, 0.03, 0.17);
            leftEye.add(leftHighlight);
            
            leftEye.position.set(-0.28, 1.95, 0.7);
            avatar.add(leftEye);
            
            // Right Eye (mirror)
            rightEye = leftEye.clone();
            rightEye.position.set(0.28, 1.95, 0.7);
            avatar.add(rightEye);
        }
        
        function createEyebrows() {
            const browGeometry = new THREE.BoxGeometry(0.25, 0.06, 0.03);
            const browMaterial = new THREE.MeshPhongMaterial({ color: 0x3d2817 });
            
            leftEyebrow = new THREE.Mesh(browGeometry, browMaterial);
            leftEyebrow.position.set(-0.28, 2.15, 0.78);
            leftEyebrow.rotation.z = -0.1;
            avatar.add(leftEyebrow);
            
            rightEyebrow = new THREE.Mesh(browGeometry, browMaterial);
            rightEyebrow.position.set(0.28, 2.15, 0.78);
            rightEyebrow.rotation.z = 0.1;
            avatar.add(rightEyebrow);
        }
        
        function createNose() {
            const nose = new THREE.Group();
            
            // Nose bridge
            const bridgeGeometry = new THREE.BoxGeometry(0.12, 0.4, 0.12);
            const noseMaterial = new THREE.MeshPhongMaterial({ color: 0xffb899 });
            const bridge = new THREE.Mesh(bridgeGeometry, noseMaterial);
            nose.add(bridge);
            
            // Nose tip
            const tipGeometry = new THREE.SphereGeometry(0.1, 16, 16);
            const tip = new THREE.Mesh(tipGeometry, noseMaterial);
            tip.position.set(0, -0.15, 0.08);
            tip.scale.set(1, 0.8, 1.2);
            nose.add(tip);
            
            // Nostrils
            const nostrilGeometry = new THREE.SphereGeometry(0.03, 8, 8);
            const nostrilMaterial = new THREE.MeshPhongMaterial({ color: 0x554433 });
            const leftNostril = new THREE.Mesh(nostrilGeometry, nostrilMaterial);
            leftNostril.position.set(-0.05, -0.2, 0.12);
            leftNostril.scale.set(1, 0.6, 1);
            nose.add(leftNostril);
            
            const rightNostril = leftNostril.clone();
            rightNostril.position.set(0.05, -0.2, 0.12);
            nose.add(rightNostril);
            
            nose.position.set(0, 1.8, 0.75);
            avatar.add(nose);
        }
        
        function createMouth() {
            mouth = new THREE.Group();
            
            // Upper lip
            const upperLipGeometry = new THREE.TorusGeometry(0.2, 0.04, 8, 16, Math.PI);
            const lipMaterial = new THREE.MeshPhongMaterial({ color: 0xcc6666 });
            const upperLip = new THREE.Mesh(upperLipGeometry, lipMaterial);
            upperLip.rotation.x = Math.PI;
            mouth.add(upperLip);
            
            // Lower lip
            const lowerLipGeometry = new THREE.TorusGeometry(0.2, 0.045, 8, 16, Math.PI);
            const lowerLip = new THREE.Mesh(lowerLipGeometry, lipMaterial);
            lowerLip.position.y = -0.05;
            mouth.add(lowerLip);
            
            mouth.position.set(0, 1.45, 0.8);
            avatar.add(mouth);
        }
        
        function createEars() {
            const earGeometry = new THREE.SphereGeometry(0.15, 16, 16);
            earGeometry.scale(0.6, 1, 0.4);
            const earMaterial = new THREE.MeshPhongMaterial({ color: 0xffc9a3 });
            
            const leftEar = new THREE.Mesh(earGeometry, earMaterial);
            leftEar.position.set(-0.82, 1.8, 0);
            leftEar.rotation.z = -0.3;
            avatar.add(leftEar);
            
            const rightEar = new THREE.Mesh(earGeometry, earMaterial);
            rightEar.position.set(0.82, 1.8, 0);
            rightEar.rotation.z = 0.3;
            avatar.add(rightEar);
        }
        
        function createBody() {
            // Torso with shirt
            const torsoGeometry = new THREE.CylinderGeometry(0.5, 0.65, 1.2, 16);
            const shirtMaterial = new THREE.MeshPhongMaterial({ 
                color: 0x4a90e2,
                shininess: 5
            });
            body = new THREE.Mesh(torsoGeometry, shirtMaterial);
            body.position.y = 0.4;
            body.castShadow = true;
            avatar.add(body);
            
            // Collar
            const collarGeometry = new THREE.TorusGeometry(0.32, 0.05, 8, 16);
            const collarMaterial = new THREE.MeshPhongMaterial({ color: 0x3a7bc2 });
            const collar = new THREE.Mesh(collarGeometry, collarMaterial);
            collar.position.y = 0.95;
            collar.rotation.x = Math.PI / 2;
            avatar.add(collar);
        }
        
        function createArms() {
            const armMaterial = new THREE.MeshPhongMaterial({ color: 0x4a90e2 });
            const handMaterial = new THREE.MeshPhongMaterial({ color: 0xffc9a3 });
            
            // Left Arm
            leftArm = new THREE.Group();
            
            const upperArmGeometry = new THREE.CylinderGeometry(0.12, 0.11, 0.7, 12);
            const leftUpperArm = new THREE.Mesh(upperArmGeometry, armMaterial);
            leftUpperArm.position.y = -0.35;
            leftArm.add(leftUpperArm);
            
            const forearmGeometry = new THREE.CylinderGeometry(0.11, 0.09, 0.6, 12);
            const leftForearm = new THREE.Mesh(forearmGeometry, armMaterial);
            leftForearm.position.y = -1;
            leftArm.add(leftForearm);
            
            const handGeometry = new THREE.SphereGeometry(0.12, 12, 12);
            handGeometry.scale(1, 1.2, 0.7);
            const leftHand = new THREE.Mesh(handGeometry, handMaterial);
            leftHand.position.y = -1.35;
            leftArm.add(leftHand);
            
            leftArm.position.set(-0.62, 0.8, 0);
            leftArm.rotation.z = 0.4;
            avatar.add(leftArm);
            
            // Right Arm
            rightArm = leftArm.clone();
            rightArm.position.set(0.62, 0.8, 0);
            rightArm.rotation.z = -0.4;
            avatar.add(rightArm);
        }
        
        function createHair() {
            const hairGroup = new THREE.Group();
            
            // Main hair volume
            const mainHairGeometry = new THREE.SphereGeometry(0.9, 24, 24, 0, Math.PI * 2, 0, Math.PI * 0.6);
            const hairMaterial = new THREE.MeshPhongMaterial({ 
                color: 0x2d1810,
                shininess: 20
            });
            const mainHair = new THREE.Mesh(mainHairGeometry, hairMaterial);
            mainHair.position.y = 2.3;
            hairGroup.add(mainHair);
            
            // Hair fringe
            for (let i = 0; i < 5; i++) {
                const fringeGeometry = new THREE.SphereGeometry(0.15, 12, 12);
                const fringe = new THREE.Mesh(fringeGeometry, hairMaterial);
                fringe.position.set((i - 2) * 0.2, 2.15, 0.7);
                fringe.scale.set(1, 1.5, 0.8);
                hairGroup.add(fringe);
            }
            
            avatar.add(hairGroup);
        }
        
        function changeGesture(gesture) {
            currentGesture = gesture;
            updateGesture();
        }
        
        function updateGesture() {
            // Reset to neutral
            leftArm.rotation.set(0, 0, 0.4);
            rightArm.rotation.set(0, 0, -0.4);
            head.rotation.set(0, 0, 0);
            mouth.rotation.set(0, 0, 0);
            mouth.scale.set(1, 1, 1);
            leftEyebrow.rotation.set(0, 0, -0.1);
            rightEyebrow.rotation.set(0, 0, 0.1);
            leftEyebrow.position.y = 2.15;
            rightEyebrow.position.y = 2.15;
            
            switch(currentGesture) {
                case 'happy':
                    mouth.rotation.x = -0.3;
                    mouth.scale.y = 1.3;
                    leftEye.scale.y = 0.7;
                    rightEye.scale.y = 0.7;
                    leftEyebrow.rotation.z = -0.2;
                    rightEyebrow.rotation.z = 0.2;
                    break;
                    
                case 'sad':
                    mouth.rotation.x = 0.3;
                    mouth.scale.y = 0.8;
                    leftEyebrow.rotation.z = 0.2;
                    rightEyebrow.rotation.z = -0.2;
                    leftEyebrow.position.y = 2.1;
                    rightEyebrow.position.y = 2.1;
                    head.rotation.z = 0.1;
                    break;
                    
                case 'thinking':
                    head.rotation.set(-0.15, 0.3, 0);
                    rightArm.rotation.set(-0.8, 0, -1.7);
                    leftEyebrow.position.y = 2.18;
                    rightEyebrow.position.y = 2.18;
                    break;
                    
                case 'excited':
                    mouth.scale.set(1.4, 1.4, 1.4);
                    leftEye.scale.set(1.3, 1.3, 1.3);
                    rightEye.scale.set(1.3, 1.3, 1.3);
                    leftArm.rotation.z = 1.2;
                    rightArm.rotation.z = -1.2;
                    leftEyebrow.position.y = 2.2;
                    rightEyebrow.position.y = 2.2;
                    break;
                    
                case 'surprised':
                    mouth.scale.set(1.2, 1.5, 1.2);
                    leftEye.scale.set(1.4, 1.4, 1.4);
                    rightEye.scale.set(1.4, 1.4, 1.4);
                    leftEyebrow.position.y = 2.25;
                    rightEyebrow.position.y = 2.25;
                    break;
                    
                case 'waving':
                    rightArm.rotation.set(0.5, 0, -2.2);
                    head.rotation.y = -0.2;
                    mouth.rotation.x = -0.2;
                    break;
                    
                case 'pointing':
                    rightArm.rotation.set(0, -0.5, -1.5);
                    head.rotation.y = -0.3;
                    break;
            }
        }
        
        function startTalking() {
            isTalking = true;
        }
        
        function stopTalking() {
            isTalking = false;
        }
        
        function respond(type) {
            const gestureMap = {
                'greeting': 'waving',
                'explaining': 'pointing',
                'thinking': 'thinking',
                'happy': 'happy',
                'excited': 'excited'
            };
            
            changeGesture(gestureMap[type] || 'neutral');
            startTalking();
            setTimeout(() => stopTalking(), 3000);
        }
        
        function animate() {
            requestAnimationFrame(animate);
            animationFrame++;
            
            // Talking animation
            if (isTalking) {
                mouth.scale.y = 1 + Math.abs(Math.sin(animationFrame * 0.2)) * 0.4;
            }
            
            // Natural breathing
            body.scale.y = 1 + Math.sin(animationFrame * 0.03) * 0.015;
            
            // Realistic blinking
            if (animationFrame % 180 < 8) {
                leftEye.scale.y = 0.1;
                rightEye.scale.y = 0.1;
            } else if (animationFrame % 180 < 12) {
                leftEye.scale.y = 0.5;
                rightEye.scale.y = 0.5;
            } else {
                if (currentGesture !== 'happy') {
                    leftEye.scale.y = 1;
                    rightEye.scale.y = 1;
                }
            }
            
            // Waving animation
            if (currentGesture === 'waving') {
                rightArm.rotation.x = 0.5 + Math.sin(animationFrame * 0.15) * 0.4;
            }
            
            // Subtle idle movement
            avatar.position.y = Math.sin(animationFrame * 0.02) * 0.02;
            head.rotation.y += Math.sin(animationFrame * 0.015) * 0.0003;
            
            renderer.render(scene, camera);
        }
        
        window.addEventListener('resize', function() {
            camera.aspect = window.innerWidth / 600;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, 600);
        });
        
        init();
    </script>
</body>
</html>
        """
    
    def display(self):
        """Display the avatar in Colab"""
        display(HTML(self._get_html()))
    
    def respond(self, response_type):
        """
        Trigger avatar response based on chatbot output
        
        Args:
            response_type: 'greeting', 'happy', 'thinking', 'explaining', 
                          'confused', 'sad', 'excited', 'surprised'
        """
        print(f"Avatar responding with: {response_type}")
        js_code = f"""
        <script>
            if (typeof respond !== 'undefined') {{
                respond('{response_type}');
            }}
        </script>
        """
        display(HTML(js_code))

# Usage in Google Colab:
# avatar = RealisticHumanAvatar()
# avatar.display()
# 
# Integration with chatbot:
# avatar.respond('greeting')  # For hello/hi
# avatar.respond('thinking')  # When processing
# avatar.respond('explaining') # When giving answers
# avatar.respond('happy')     # For positive responses
# avatar.respond('excited')   # For enthusiastic responses
# avatar.respond('surprised') # For unexpected queries
