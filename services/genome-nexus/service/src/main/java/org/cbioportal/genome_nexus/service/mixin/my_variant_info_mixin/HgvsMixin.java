package org.cbioportal.genome_nexus.service.mixin.my_variant_info_mixin;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

@JsonIgnoreProperties(ignoreUnknown = true)
public class HgvsMixin
{
    @JsonProperty(value = "coding", required = true)
    private String coding;

    @JsonProperty(value = "genomic", required = true)
    private String genomic;
}