import vcf

def extract_genes_clinvar(gene_info):
    genes = []
    for item in gene_info:
        gene_names = item.split('|')
        for gene in gene_names:
            if not gene.isdigit():
                genes.append(gene.split(':')[0])
    unique_genes = list(set(genes))
    return unique_genes


def extract_mc_clinvar(mc_info):
    mc = []
    for item in mc_info:
        mc.append(item.split('|')[-1])
    return ' and '.join(mc)


# ClinVar SV data
def get_clinval_sv_data(sv_file):
    vcf_reader = vcf.Reader(filename=sv_file)
    vcfINFO = {"Duplication": [], "Insertion": [], 'Microsatellite': [], 'Inversion': [], 'Deletion': []}
    clnsig_label = {"Pathogenic":"Pathogenic", "Pathogenic/Likely_pathogenic":"Pathogenic", "Benign":"Benign", "Benign/Likely_benign":"Benign"}
    for record in vcf_reader:
        temp = []
        
        try:
            clnsig =  "".join(str(i) for i in record.INFO['CLNSIG'])
            variation_type = record.INFO['CLNVC']
            chrom = record.CHROM
            pos = record.POS
            ref = record.REF
            alt = "".join(str(i) for i in record.ALT)

            try:
                mc =  extract_mc_clinvar(record.INFO['MC'])
            except:
                mc = 'None'
            genes = extract_genes_clinvar([record.INFO['GENEINFO']])
            end = pos + (len(ref) if len(ref) > len(alt) else len(alt))

            if clnsig in clnsig_label:
                clnsig =  clnsig_label[clnsig]
                temp = [chrom, pos, end, mc, genes, clnsig]
                if variation_type in vcfINFO:
                    vcfINFO[variation_type].append(temp)
            
            elif clnsig == 'Likely_benign' and variation_type == 'Inversion':
                temp = [chrom, pos, end, mc, genes, 'Benign']
                vcfINFO[variation_type].append(temp)
            elif clnsig == 'Likely_pathogenic' and variation_type == 'Inversion':
                temp = [chrom, pos, end, mc, genes, 'Benign']
                vcfINFO[variation_type].append(temp)
        except:
            pass
    return vcfINFO

# dbVar SV data
def get_dbvar_sv_data(sv_file):
    vcf_reader = vcf.Reader(filename=sv_file)
    vcfINFO = {"Duplication": [], "Insertion": [], 'Microsatellite': [], 'Inversion': [], 'Deletion': []}
    for record in vcf_reader:
        tmp = []
        try:
            chrom = record.CHROM
            pos = record.POS
            end = record.INFO['END']    
            mc = "".join(str(i) for i in record.INFO['ExonicFunc.refGene'])
            gene = "".join(str(i) for i in record.INFO['Gene.refGene']).split('\\x3b')
            sv_type = record.INFO["SVTYPE"]
            if 'CLNSIG' in record.INFO:
                clnsig = record.INFO['CLNSIG']
            elif 'CLINICAL_SIGNIFICANCE' in record.INFO:
                clnsig = record.INFO['CLINICAL_SIGNIFICANCE']
            elif 'CLINICAL_ASSERTION' in record.INFO:
                clnsig = record.INFO['CLINICAL_ASSERTION']
            else:
                clnsig = 'unknown'
            clnsig = "".join(str(i) for i in clnsig)
            tmp = [chrom, pos, end, mc, gene, clnsig]
            if clnsig != 'unknown' and tmp:                
                if sv_type in vcfINFO:
                    if sv_type in ['Deletion', 'Duplication']:
                        if end - pos >= 50:
                            vcfINFO[sv_type].append(tmp)
                    else:
                        vcfINFO[sv_type].append(tmp)
        except:
            pass           
        
    return vcfINFO 



def read_vcf(sv_file):
    vcf_reader = vcf.Reader(filename=sv_file)
    vcfINFO = {"Duplication": [], "Insertion": [], 'Microsatellite': [], 'Inversion': [], 'Deletion': []}
    for record in vcf_reader:
        tmp = []
        if True:
            chrom = record.CHROM
            pos = record.POS
            try:
                end = record.INFO['END'] 
            except:
                ref = record.REF
                alt =  "".join(str(i) for i in record.ALT)
                if len(ref) > len(alt):
                    end = pos + len(ref)
                else:
                    end = pos + len(alt)
                
            try:  
                mc = "".join(str(i) for i in record.INFO['ExonicFunc.refGene'])
                gene = "".join(str(i) for i in record.INFO['Gene.refGene']).split('\\x3b')
            except:
                try:
                    mc =  extract_mc_clinvar(record.INFO['MC'])
                    gene = extract_genes_clinvar([record.INFO['GENEINFO']])
                except:
                    mc = 'unknown'
                    gene = 'None'
            
            try:
                sv_type = record.INFO["SVTYPE"]
            except:
                sv_type = record.INFO['CLNVC']
            # if not mc or  not gene:
            #     print(f'Molecular consequence or gene list not found in record {chrom}:{pos}-{end}')
            #     continue
            tmp = [chrom, pos, end, mc, gene]
           
            if sv_type in vcfINFO:      
                vcfINFO[sv_type].append(tmp)
        # except:
        #     pass           
        
    return vcfINFO 