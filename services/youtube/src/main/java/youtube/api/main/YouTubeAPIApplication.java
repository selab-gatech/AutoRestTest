package youtube.api.main;

import org.springframework.stereotype.Component;
import youtube.api.resources.YouTubeResource;

import javax.ws.rs.ApplicationPath;
import javax.ws.rs.core.Application;
import java.text.ParseException;
import java.util.HashSet;
import java.util.Set;

@Component
@ApplicationPath("/api")
public class YouTubeAPIApplication extends Application {

    private Set<Object> singletons = new HashSet<>();
    private Set<Class<?>> classes = new HashSet<>();

    public YouTubeAPIApplication() throws ParseException {
        singletons.add(YouTubeResource.getInstance());
    }

    @Override
    public Set<Class<?>> getClasses() {
        return classes;
    }

    @Override
    public Set<Object> getSingletons() {
        return singletons;
    }
    
}
