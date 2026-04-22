import React, { useEffect, useRef, useState } from 'react';

const ProteinViewer = ({ pdbId, hotspots }) => {
  const viewerRef = useRef(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!pdbId || !viewerRef.current || !window.$3Dmol) return;

    setLoading(true);
    let viewer = window.$3Dmol.createViewer(viewerRef.current, {
      backgroundColor: 'rgba(0,0,0,0)'
    });

    const renderProtein = async () => {
      try {
        const res = await fetch(`https://files.rcsb.org/download/${pdbId.toUpperCase()}.pdb`);
        if (!res.ok) throw new Error("Network response was not ok");
        const pdbData = await res.text();
        
        viewer.addModel(pdbData, "pdb");
        
        // BASE STYLE: Using surface for a 3D organic "realistic" look
        // Use a high-contrast color scheme so the full protein is readable on a dark UI.
        viewer.setStyle({}, { cartoon: { color: 'spectrum', opacity: 0.95 } });

        // HOTSPOT STYLE: Glow-like highlight
        if (hotspots && hotspots.length > 0) {
          hotspots.forEach(hotspot => {
            // Add a vibrant sphere for the specific residue
            viewer.addStyle(
              { resi: hotspot.node_idx },
              { sphere: { color: '#FF5E00', scale: 1.5 } } // Sunset orange hotspots
            );
            // Highlight the exact sticks for detail
            viewer.addStyle(
              { resi: hotspot.node_idx },
              { stick: { color: '#FF5E00', thickness: 0.3 } }
            );
          });
        }
        
        viewer.zoomTo();
        viewer.render();
      } catch (err) {
        console.error("Failed to render protein:", err);
      } finally {
        setLoading(false);
      }
    };

    renderProtein();

    return () => {
      if (viewer) {
        viewer.clear();
      }
      if (viewerRef.current) {
        viewerRef.current.innerHTML = '';
      }
    };
  }, [pdbId, hotspots]);

  return (
    <div className="relative w-full h-[500px] glass-card bg-slate-900/60 rounded-xl border border-slate-700 overflow-hidden flex items-center justify-center">
      {loading && <div className="absolute z-10 text-slate-400 animate-pulse">Loading structure...</div>}
      <div ref={viewerRef} className="w-full h-full relative z-0" />
    </div>
  );
};

export default ProteinViewer;
