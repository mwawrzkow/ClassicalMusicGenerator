import{c as V,a as s,_ as N,b as q,d as U,e as h}from"./Cq73w6-n.js";import{_ as x,a as g}from"./DJuun8Rc.js";import{e as $,s as y,f as z,w as m,o as B,a as C,b as n,x as t,d as G,_ as S}from"./CdEF7n17.js";import"./WwBUNse5.js";import"./BRGGRA3v.js";const T={data(){return{items:["GAN","RNN"],selected:"GAN",name:""}},methods:{async submit(i){var d=JSON.stringify(i.data);(await fetch("/train",{method:"POST",headers:{"Content-Type":"application/json"},body:d})).status===200&&this.$router.push("/outputlogs")}}},j=$({...T,__name:"train",setup(i){const d=["GAN","RNN"],_=V({model_name:s().oneOf(d).required(),num_generations:s().matches(/^\d+$/).required(),midi_files_dir:s(),num_sequences:s().matches(/^\d+$/).required(),num_epochs:s().matches(/^\d+$/).required(),batch_size:s().matches(/^\d+$/).required(),num_files:s().matches(/^\d+$/).required(),temperature:s().matches(/^\d+$/).required()}),o=y({model_name:d[0],num_generations:10,midi_files_dir:"",num_sequences:100,num_epochs:10,batch_size:10,num_files:10,temperature:1});return(r,e)=>{const p=N,l=q,u=U,f=g,c=h,b=x;return B(),z(b,{class:"text-center h-screen flex flex-col justify-center"},{default:m(()=>[e[9]||(e[9]=C("h1",null,"Configure model",-1)),n(c,{schema:t(_),state:t(o),onSubmit:r.submit},{default:m(()=>[n(l,{label:"Model name",name:"model_name"},{default:m(()=>[n(p,{options:r.items,modelValue:t(o).model_name,"onUpdate:modelValue":e[0]||(e[0]=a=>t(o).model_name=a),id:"model_name"},null,8,["options","modelValue"])]),_:1}),n(l,{label:"Number of generations",name:"num_generations"},{default:m(()=>[n(u,{modelValue:t(o).num_generations,"onUpdate:modelValue":e[1]||(e[1]=a=>t(o).num_generations=a),id:"name"},null,8,["modelValue"])]),_:1}),n(l,{label:"Midi files directory",name:"midi_files_dir"},{default:m(()=>[n(u,{modelValue:t(o).midi_files_dir,"onUpdate:modelValue":e[2]||(e[2]=a=>t(o).midi_files_dir=a),id:"midi_files_dir"},null,8,["modelValue"])]),_:1}),n(l,{label:"Number of sequences",name:"num_sequences"},{default:m(()=>[n(u,{modelValue:t(o).num_sequences,"onUpdate:modelValue":e[3]||(e[3]=a=>t(o).num_sequences=a),id:"num_sequences"},null,8,["modelValue"])]),_:1}),n(l,{label:"Number of epochs",name:"num_epochs"},{default:m(()=>[n(u,{modelValue:t(o).num_epochs,"onUpdate:modelValue":e[4]||(e[4]=a=>t(o).num_epochs=a),id:"num_epochs"},null,8,["modelValue"])]),_:1}),n(l,{label:"Batch size",name:"batch_size"},{default:m(()=>[n(u,{modelValue:t(o).batch_size,"onUpdate:modelValue":e[5]||(e[5]=a=>t(o).batch_size=a),id:"batch_size"},null,8,["modelValue"])]),_:1}),n(l,{label:"Number of files",name:"num_files"},{default:m(()=>[n(u,{modelValue:t(o).num_files,"onUpdate:modelValue":e[6]||(e[6]=a=>t(o).num_files=a),id:"num_files"},null,8,["modelValue"])]),_:1}),n(l,{label:"Temperature",name:"temperature"},{default:m(()=>[n(u,{modelValue:t(o).temperature,"onUpdate:modelValue":e[7]||(e[7]=a=>t(o).temperature=a),id:"temperature"},null,8,["modelValue"])]),_:1}),n(f,{type:"submit"},{default:m(()=>e[8]||(e[8]=[G("Generate")])),_:1})]),_:1},8,["schema","state","onSubmit"])]),_:1})}}}),k=S(j,[["__scopeId","data-v-1ec131b8"]]);export{k as default};
