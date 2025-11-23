import React, { useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';

const SpinningTire = ({ compound, wear }) => {
  const meshRef = useRef();
  
  // Rotate the tire
  useFrame((state, delta) => {
    meshRef.current.rotation.x += delta * 2; // Spin speed
    meshRef.current.rotation.y += delta * 0.5;
  });

  // Color logic
  const getColor = () => {
    if (compound === 'SOFT') return '#FF1801';
    if (compound === 'MEDIUM') return '#FFD700';
    return '#FFFFFF';
  };

  return (
    <mesh ref={meshRef} rotation={[1.5, 0, 0]}>
      {/* A Torus looks more like a tire than a cylinder */}
      <torusGeometry args={[2, 0.8, 16, 50]} /> 
      <meshStandardMaterial 
        color={getColor()} 
        wireframe={wear < 20} // Show wireframe if tire is dead
        roughness={0.8}
      />
    </mesh>
  );
};

const Tire3D = ({ compound, wear }) => {
  return (
    <div className="h-[200px] w-full">
      <Canvas>
        <ambientLight intensity={0.5} />
        <spotLight position={[10, 10, 10]} angle={0.15} penumbra={1} />
        <pointLight position={[-10, -10, -10]} />
        <SpinningTire compound={compound} wear={wear} />
        <OrbitControls enableZoom={false} />
      </Canvas>
    </div>
  );
};

export default Tire3D;