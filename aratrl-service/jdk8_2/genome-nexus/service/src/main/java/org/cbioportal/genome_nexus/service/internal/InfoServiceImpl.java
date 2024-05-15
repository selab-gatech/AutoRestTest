package org.cbioportal.genome_nexus.service.internal;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.PropertySource;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;


@Configuration
@PropertySource("classpath:/maven.properties")
@Service
public class InfoServiceImpl {
    @Autowired
    private Version version;

    public Version getVersion() {
        return this.version;
    }

    @Component
    public static class Version {
        private String version;

        @Autowired
        Version(@Value("${project.version}") final String version) {
            this.version = version;
        }

        public String getVersion() {
            return this.version;
        }
    }

}