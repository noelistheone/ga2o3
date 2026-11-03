"""Traceable temperature-Hall curve reads; do not treat pixels as specimens."""
from pathlib import Path
import os,sys,json,hashlib
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
from PIL import Image,ImageDraw
import fitz
import phase2_energy as energy
from csv_io_v2 import read_result_csv

OUT=energy.BASE/"phase8_process_temperature/extraction_v2"
NET=ROOT.parent/"Ga2O3-Net"
SOURCES={
    "xiang2022":dict(doi="10.1007/s40843-022-2167-x",
        path=NET/"extra-paper/doping_extracted/doping/si/s40843-022-2167-x.pdf",page=2),
    "mccandless2022":dict(doi="10.1063/5.0101132",
        path=NET/"extra-paper/doping_extracted/doping/si/072108_1_online.pdf",page=4),
    "mauze2020":dict(doi="10.1063/5.0027870",
        path=NET/"extra-paper/doping_extracted/doping/Sn/222102_1_online.pdf",page=4),
}
PANELS={
    "A":dict(x0=141.,x1=815.,y0=10,y1=554,yref=98.,logref=16.,pixels_per_decade=162.,
             legend=[550,10,815,130]),
    "B":dict(x0=979.,x1=1654.,y0=10,y1=554,yref=42.,logref=17.,pixels_per_decade=182.,
             legend=[1380,10,1654,130]),
    "C":dict(x0=141.,x1=815.,y0=685,y1=1229,yref=875.,logref=17.,pixels_per_decade=273.,
             legend=[550,685,815,805]),
    "D":dict(x0=979.,x1=1654.,y0=685,y1=1229,yref=921.,logref=17.,pixels_per_decade=237.,
             legend=[1380,685,1654,805]),
}

def samples():
    d=[]
    for j,s in enumerate("ABCD"):
        C=[1.93e16,9.5e16,5.4e17,5.62e17][j]
        d.append(dict(sample_id=SOURCES["xiang2022"]["doi"]+"::"+s,source="xiang2022",
            family=SOURCES["xiang2022"]["doi"],sample_label=s,element="Si",
            incorporated_cm3=C,chemical_lower_cm3=max(C-1e16,1),chemical_upper_cm3=C+1e16,
            chemical_bound_kind="source reports SIMS error below1e16; conservative absolute bound, not95%CI",
            reported_RT_nH_cm3=[1.83e16,8.64e16,3.35e17,6.55e17][j],growth_T_C=830.,
            total_pressure_Torr=60.,oxygen_partial_pressure_status="unreported",
            TEGa_sccm=[100,90,70,100][j],O2_sccm=[400,400,1000,400][j],
            SiH4_sccm=[1,5,5,10][j],growth_hours=[3,3,2,3][j],
            thickness_um=[1.64,1.37,.93,1.68][j],
            H_detected_cm3=[None,None,1.8e17,8.83e16][j],
            H_status="below unspecified detection limit" if j<2 else "detected",
            C_detected_cm3=7.52e18 if s=="D" else None,
            C_status="surface accumulation, nonuniform" if s=="D" else "below unspecified detection limit",
            method="MOCVD",substrate="Fe-doped semi-insulating beta(010)",
            contact_anneal="470C, N2,1min",Hall_instrument="PPMS, source capability2-400K,+/-9T",
            temperature_trace="Fig3",chemical_locator="p749 text/Fig2",room_temperature_locator="p749 text",
            independence="same physical specimen as existing dataset_v1 row, new curve reads only"))
    for j,s in enumerate(("1","2","3")):
        d.append(dict(sample_id=SOURCES["mccandless2022"]["doi"]+"::"+s,source="mccandless2022",
            family=SOURCES["mccandless2022"]["doi"],sample_label=s,element="Si",
            incorporated_cm3=[1.30e18,1.09e18,1.4e17][j],
            chemical_lower_cm3=None,chemical_upper_cm3=None,chemical_bound_kind="experimental SIMS error unreported",
            reported_RT_nH_cm3=[8.16e17,4.38e17,1.11e17][j],reported_RT_mu=[107.4,87.5,128.6][j],
            reported_peak_mu=[219.9,189.6,390.2][j],reported_peak_T_K=[115,110,97][j],
            growth_T_C=[523,502,523][j],Ga_flux_atoms_nm2s=[1.1,.7,1.1][j],
            O_flux_atoms_nm2s=[2.,1.4,2.][j],dopant_cell_T_C=[900,905,890][j],
            method="PA-MBE with modified Si source",growth_rate_nm_h=[93.,60.,93.][j],
            oxygen_partial_pressure_status="active oxygen flux not equilibrium chemical potential",
            substrate="Fe-doped semi-insulating beta(010)",
            temperature_trace="Fig4a carrier;Fig4e mobility",chemical_locator="TableI/Fig4b",
            room_temperature_locator="TableI",
            independence="same physical specimen as existing dataset_v1 row, new curve reads only"))
    return d

def freeze():
    payload=dict(version=2,created_utc=energy.now(),code_sha256=energy.sha(Path(__file__)),
        sources={k:dict(doi=v["doi"],path=str(v["path"].resolve()),sha256=energy.sha(v["path"]),page_index=v["page"])
                 for k,v in SOURCES.items()},
        samples=samples(),xiang_axes=PANELS,
        xiang_read_grid_inverseK_times1000=list(range(3,26)),xiang_window_radius_px=11,
        xiang_pixel_rule="RGB all<100,range<35; inside axes25px inset; legend removed; nearest black column, median black y; no red fit pixels used",
        xiang_ambiguous_rule="retain missing if black-y span>60px, or no black pixel; no fit/interpolation fill",
        xiang_axis_uncertainty_px=2.,xiang_scope="processed published measured trace reads, not raw point-count reconstruction",
        mccandless_carrier_axes=dict(T0_x=90.658,T300_x=155.824,ylog18=573.4787,ylog13=634.8707),
        mccandless_mobility_axes=dict(T0_x=473.849,T300_x=541.023,y0=634.8707,y400=571.6107),
        vector_center_rule="bounding-box center; red plus two paths combined within.02pt; symbol extent retained as reading bound",
        repeated_axes="use only Fig4a, not duplicate Fig4c; curve+tableRT are not independent specimens",
        temperature_scope="retain all visible readings; lowT impurity/interface warnings flagged, not discarded",
        prediction_goal="paired incorporated-dose+process+T to Hall carrier; later same-specimen mobility",
        source_split="whole DOI and specimen groups; same7previous specimens/2DOI, not7new experiments",
        excluded_pair="MauzeFig5 lacks explicit same-specimen chemical SIMS; conditional curve only, no use of fittedND as chemical dose",
        observation_caveat="Hall apparent density may involve scattering factors and parallel channels; physical carrier!=Hall label automatically",
        fitting="none in extraction; future learning protocol must freeze after these inputs are audited",
        grade="retrospective source-exposed digitization, not blind validation",
        old_manifest_sha256=energy.sha(energy.BASE/"artifact_manifest_phase7.json"))
    p=OUT/"extraction_protocol.json"
    if p.exists():
        old=json.loads(p.read_text())
        for k in ("code_sha256","sources","samples","xiang_axes"):assert old[k]==energy.clean(payload[k]),k
    else:energy.write(p,payload)
    energy.csvwrite(OUT/"paired_specimens.csv",samples())
    return payload

def extract_xiang():
    source=SOURCES["xiang2022"];doc=fitz.open(source["path"]);page=doc[source["page"]]
    info=[v for v in page.get_images(full=True) if v[2]==1671 and v[3]==1320]
    assert len(info)==1
    blob=doc.extract_image(info[0][0]);path=OUT/("xiang_Fig3_original."+blob["ext"]);path.write_bytes(blob["image"])
    im=Image.open(path).convert("RGB");rgb=np.asarray(im).astype(int)
    black=(rgb.max(axis=2)<100)&(np.ptp(rgb,axis=2)<35)
    overlay=im.copy();draw=ImageDraw.Draw(overlay);out=[];attempts=[]
    for label,p in PANELS.items():
        mask=black.copy();ys,xs=np.indices(mask.shape)
        mask &= (xs>=p["x0"]+25)&(xs<=p["x1"]-25)&(ys>=p["y0"]+25)&(ys<=p["y1"]-25)
        lx,ly,rx,ry=p["legend"];mask[ly:ry+1,lx:rx+1]=False
        for target in range(3,26):
            xp=p["x0"]+(target-2)/24*(p["x1"]-p["x0"])
            yy,xx=np.nonzero(mask&(abs(xs-xp)<=11))
            status="read"
            if not len(xx):status="missing_black_measured_marker"
            elif np.ptp(yy)>60:status="ambiguous_black_features"
            attempt=dict(source="xiang2022",sample=label,target_inverseK1000=target,target_pixel_x=xp,
                         status=status,black_pixels=len(xx))
            attempts.append(attempt)
            if status!="read":continue
            nearest=int(xx[np.argmin(abs(xx-xp))]);close=abs(xx-nearest)<=1
            px=float(np.median(xx[close]));py=float(np.median(yy[close]))
            inv=2+24*(px-p["x0"])/(p["x1"]-p["x0"]);T=1000/inv
            y=p["logref"]-(py-p["yref"])/p["pixels_per_decade"]
            # Full local glyph extent/window range and shared axis reading uncertainty,
            # not experimental uncertainty or a calibrated distribution.
            ylo=p["logref"]-(float(yy.max())+2-p["yref"])/p["pixels_per_decade"]
            yhi=p["logref"]-(float(yy.min())-2-p["yref"])/p["pixels_per_decade"]
            invlo=2+24*(float(xx.min())-2-p["x0"])/(p["x1"]-p["x0"])
            invhi=2+24*(float(xx.max())+2-p["x0"])/(p["x1"]-p["x0"])
            out.append(dict(read_id=f"xiang2022::{label}::x{target:02}",
                sample_id=source["doi"]+"::"+label,family=source["doi"],source="xiang2022",
                endpoint="log10_nH_cm3",T_K=T,observed=y,reading_lower=ylo,reading_upper=yhi,
                T_lower_K=1000/invhi,T_upper_K=1000/invlo,pixel_x=px,pixel_y=py,
                extraction="black measured trace raster;fixed inverseT grid",raw_instrument=False,
                low_temperature_warning=bool(T<100),coordinate_status=status))
            draw.ellipse((px-4,py-4,px+4,py+4),outline="lime",width=2)
    overlay.save(OUT/"xiang_measured_trace_overlay.png")
    return out,attempts

def extract_mccandless(protocol):
    s=SOURCES["mccandless2022"];page=fitz.open(s["path"])[s["page"]]
    allrows=[];objects=[]
    for endpoint,axis,xrange in (("log10_nH_cm3",protocol["mccandless_carrier_axes"],(90.,159.)),
                                ("muH_cm2Vs",protocol["mccandless_mobility_axes"],(473.,542.))):
        points={"1":[],"2":[],"3":[]}
        for i,d in enumerate(page.get_drawings()):
            fill=d["fill"];r=d["rect"]
            if fill is None or not (xrange[0]<r.x0 and r.x1<xrange[1] and 567.<r.y0<635.3):continue
            if max(fill)-min(fill)<.5:continue
            if fill[0]>.95 and fill[1]<.01 and fill[2]<.01:
                label="1";use=len(d["items"])==1 and max(r.width,r.height)<2.6
            elif fill[1]>.59 and fill[0]<.01 and fill[2]<.01:
                label="2";use=len(d["items"])==13 and 2<r.width<3.5 and 2<r.height<3.5
            elif fill[2]>.99 and fill[0]<.01 and .2<fill[1]<.3:
                label="3";use=len(d["items"])==11 and 2<r.width<3 and 2<r.height<3
            else:continue
            if use:points[label].append(dict(x=(r.x0+r.x1)/2,y=(r.y0+r.y1)/2,bbox=list(r),objects=[i]))
        for label,raw in points.items():
            merged=[]
            for point in raw:
                near=[p for p in merged if abs(p["x"]-point["x"])<.02 and abs(p["y"]-point["y"])<.02]
                if near:
                    p=near[0];p["objects"]+=point["objects"]
                    p["bbox"]=[min(p["bbox"][0],point["bbox"][0]),min(p["bbox"][1],point["bbox"][1]),
                               max(p["bbox"][2],point["bbox"][2]),max(p["bbox"][3],point["bbox"][3])]
                else:merged.append(point)
            for k,p in enumerate(sorted(merged,key=lambda q:q["x"])):
                x,y=p["x"],p["y"];box=p["bbox"]
                T=300*(x-axis["T0_x"])/(axis["T300_x"]-axis["T0_x"])
                tl=300*(box[0]-axis["T0_x"])/(axis["T300_x"]-axis["T0_x"])
                th=300*(box[2]-axis["T0_x"])/(axis["T300_x"]-axis["T0_x"])
                if endpoint=="log10_nH_cm3":
                    conv=lambda v:18-5*(v-axis["ylog18"])/(axis["ylog13"]-axis["ylog18"])
                else:
                    conv=lambda v:400*(axis["y0"]-v)/(axis["y0"]-axis["y400"])
                allrows.append(dict(read_id=f"mccandless2022::{label}::{endpoint}::{k:02}",
                    sample_id=s["doi"]+"::"+label,family=s["doi"],source="mccandless2022",
                    endpoint=endpoint,T_K=T,observed=conv(y),
                    reading_lower=conv(box[3]),reading_upper=conv(box[1]),
                    T_lower_K=tl,T_upper_K=th,pdf_x=x,pdf_y=y,object_indices=";".join(map(str,p["objects"])),
                    extraction="PDF vector glyph bounding center;symbol extent is reading bound",
                    raw_instrument=False,low_temperature_warning=bool(T<100),coordinate_status="read"))
                objects.append(dict(endpoint=endpoint,sample=label,**p))
    energy.write(OUT/"mccandless_vector_objects.json",objects)
    return allrows

def main():
    protocol=freeze();xr,attempts=extract_xiang();vr=extract_mccandless(protocol)
    rows=xr+vr;energy.csvwrite(OUT/"temperature_observations.csv",rows)
    energy.write(OUT/"xiang_read_attempts.json",attempts)
    data=read_result_csv(OUT/"temperature_observations.csv")
    specs=read_result_csv(OUT/"paired_specimens.csv")
    assert set(data.sample_id)==set(specs.sample_id)
    assert data.read_id.nunique()==len(data) and data.observed.notna().all()
    assert ((data.reading_lower<=data.observed)&(data.observed<=data.reading_upper)).all()
    assert ((data.T_lower_K<=data.T_K)&(data.T_K<=data.T_upper_K)).all()
    assert data.T_K.between(10,410).all()
    data.groupby(["source","sample_id","endpoint"]).agg(
        reads=("read_id","count"),T_min=("T_K","min"),T_max=("T_K","max"),
        y_min=("observed","min"),y_max=("observed","max")).reset_index().to_csv(OUT/"extraction_summary.csv",index=False)
    energy.write(OUT/"extraction_checks.json",dict(passed=True,readings=len(rows),
        actual_specimens=7,source_groups=2,new_specimens=0,
        xiang_requested_positions=len(attempts),xiang_measured_trace_reads=len(xr),
        failed_xiang_positions=[r for r in attempts if r["status"]!="read"],
        bounds_kind="conservative pixel/glyph/axis-reading bounds;not measured error or confidence interval",
        primary_role="inputs for next predictive model;not a new independent material validation"))
    print((OUT/"extraction_summary.csv").read_text())

if __name__=="__main__":main()
