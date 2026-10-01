/* Displays measured tokenizer output and feature changes; no browser model inference. */
(function(global){'use strict';
 global.RowEncoder={mount:function(root,data){
 root.classList.add('route-widget','row-encoder');
 root.innerHTML='<h3>Same values, different sequence</h3><p>Measured BART tokenization and pooled-feature changes. Special tokens count toward length.</p><div class="row-controls"><label>Product row <select data-row aria-label="Product row"></select></label><label>Presentation <select data-variant aria-label="Presentation"><option value="baseline">Named columns</option><option value="reordered">Reversed order</option><option value="renamed">Opaque names</option></select></label></div><pre data-text></pre><div class="token-list" tabindex="0" role="region" aria-label="Token strings and IDs"></div><output aria-live="polite"></output><p class="baseline"></p><button type="button" data-reset>Reset example</button>';
 const row=root.querySelector('[data-row]'),variant=root.querySelector('[data-variant]');
 data.rows.forEach((r,i)=>{const o=document.createElement('option');o.value=String(i);o.textContent=r.id;row.append(o);});
 function update(){const index=+row.value,v=variant.value,r=data.rows[index][v],base=data.rows[index].baseline;
 root.querySelector('[data-text]').textContent=r.text;
 const list=root.querySelector('.token-list');list.replaceChildren();
 r.tokens.forEach((token,i)=>{const span=document.createElement('span');span.className='token';span.textContent=token;const small=document.createElement('small');small.textContent=String(r.token_ids[i]);span.append(small);list.append(span);});
 root.querySelector('output').textContent=r.length+' tokens · '+r.length*r.length+' attention pairs · cosine distance '+r.cosine_distance.toFixed(6)+' · typed feature change 0';
 root.querySelector('.baseline').textContent='Baseline for this row: named columns, '+base.length+' tokens. Values and target stay fixed. The typed mapping keeps canonical field identities. Cosine distance measures feature change, not prediction quality.';
 }
 row.addEventListener('change',update);variant.addEventListener('change',update);
 root.querySelector('[data-reset]').addEventListener('click',()=>{row.value='0';variant.value='baseline';update();});update();
 }};
})(window);
